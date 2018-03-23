#include <cfloat>

#include "caffe/fast_rcnn_layers.hpp"

using std::max;
using std::min;

#define PI 3.1415926

namespace caffe {

template <typename Dtype>
__global__ void RotateROIPoolForward(const int nthreads, const Dtype* bottom_data,
    const Dtype spatial_scale, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const Dtype* bottom_rois, Dtype* top_data, int* argmax_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    // rois(oriented rectangle): [batch_idx, xmin, ymin, xmax, ymax, theta]
    bottom_rois += n * 6;
    int roi_batch_ind = bottom_rois[0];
    int roi_start_w = round(bottom_rois[1] * spatial_scale);
    int roi_start_h = round(bottom_rois[2] * spatial_scale);
    int roi_end_w = round(bottom_rois[3] * spatial_scale);
    int roi_end_h = round(bottom_rois[4] * spatial_scale);
    Dtype roi_theta = bottom_rois[5] / 180.f * PI;

    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    int roi_ctr_w = roi_start_w + roi_width * 0.5;
    int roi_ctr_h = roi_start_h + roi_height * 0.5;

    // affine matrix
    Dtype aff_mat[2][2];
    aff_mat[0][0] = static_cast<Dtype>(cos(roi_theta));
    aff_mat[0][1] = static_cast<Dtype>(sin(roi_theta));
    aff_mat[1][0] = static_cast<Dtype>(-sin(roi_theta));
    aff_mat[1][1] = static_cast<Dtype>(cos(roi_theta));

    // Force malformed ROIs to be 1x1
    Dtype bin_size_h = static_cast<Dtype>(roi_height) / static_cast<Dtype>(pooled_height);
    Dtype bin_size_w = static_cast<Dtype>(roi_width) / static_cast<Dtype>(pooled_width);
    int hstart = static_cast<int>(floor(static_cast<Dtype>(ph) * bin_size_h));
    int wstart = static_cast<int>(floor(static_cast<Dtype>(pw) * bin_size_w));
    int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1) * bin_size_h));
    int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1) * bin_size_w));

    hstart = hstart + roi_start_h;
    hend = hend + roi_start_h;
    wstart = wstart + roi_start_w;
    wend = wend + roi_start_w;

    Dtype maxval = 0;
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    int maxidx = -1;
    bottom_data += (roi_batch_ind * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        // rotate
        int r_w = round(static_cast<Dtype>(w - roi_ctr_w) * aff_mat[0][0] + 
                        static_cast<Dtype>(h - roi_ctr_h) * aff_mat[0][1]) + roi_ctr_w;
        int r_h = round(static_cast<Dtype>(w - roi_ctr_w) * aff_mat[1][0] + 
                        static_cast<Dtype>(h - roi_ctr_h) * aff_mat[1][1]) + roi_ctr_h;
        // skip if [r_w, r_h] not inside
        if ((r_w < 0) || (r_h < 0) || (r_w > (width - 1)) || (r_h > (height - 1)))
          continue;
        const int bottom_index = r_h * width + r_w;
        if (bottom_data[bottom_index] > maxval) {
          maxval = bottom_data[bottom_index];
          maxidx = bottom_index;
        }
      }
    }
    top_data[index] = maxval;
    argmax_data[index] = maxidx;
  }
}

template <typename Dtype>
void RotateROIPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int* argmax_data = max_idx_.mutable_gpu_data();
  int count = top[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  RotateROIPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, spatial_scale_, channels_, height_, width_,
      pooled_height_, pooled_width_, bottom_rois, top_data, argmax_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void RotateROIPoolBackward(const int nthreads, const Dtype* top_diff,
    const int* argmax_data, const int num_rois, const Dtype spatial_scale,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, Dtype* bottom_diff,
    const Dtype* bottom_rois) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, h, w) coords in bottom data
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    Dtype gradient = 0;
    // Accumulate gradient over all ROIs that pooled this element
    for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
      const Dtype* offset_bottom_rois = bottom_rois + roi_n * 6;
      int roi_batch_ind = offset_bottom_rois[0];
      // Skip if ROI's batch index doesn't match n
      if (n != roi_batch_ind) {
        continue;
      }

      int roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
      int roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
      int roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
      int roi_end_h = round(offset_bottom_rois[4] * spatial_scale);
      Dtype roi_theta = static_cast<Dtype>(offset_bottom_rois[5] / 180.f * PI);

      int roi_width = max(roi_end_w - roi_start_w + 1, 1);
      int roi_height = max(roi_end_h - roi_start_h + 1, 1);
      int roi_ctr_w = roi_start_w + roi_width * 0.5;
      int roi_ctr_h = roi_start_h + roi_height * 0.5;

      // get affine matrix
      Dtype aff_mat[2][2];
      aff_mat[0][0] = static_cast<Dtype>(cos(roi_theta));
      aff_mat[0][1] = static_cast<Dtype>(sin(roi_theta));
      aff_mat[1][0] = static_cast<Dtype>(-sin(roi_theta));
      aff_mat[1][1] = static_cast<Dtype>(cos(roi_theta));

      // point in polygon
      // add +-1 to make sure boundary points inside
      int pt_a_w = roi_start_w - 1;
      int pt_a_h = roi_start_h - 1;
      int pt_b_w = roi_end_w + 1;
      int pt_b_h = roi_start_h - 1;
      int pt_c_w = roi_end_w + 1;
      int pt_c_h = roi_end_h + 1;
      int pt_d_w = roi_start_w - 1;
      int pt_d_h = roi_end_h + 1;
      int r_pt_a_w = round(static_cast<Dtype>(pt_a_w - roi_ctr_w) * aff_mat[0][0] +
                           static_cast<Dtype>(pt_a_h - roi_ctr_h) * aff_mat[0][1]) + roi_ctr_w;
      int r_pt_a_h = round(static_cast<Dtype>(pt_a_w - roi_ctr_w) * aff_mat[1][0] +
                           static_cast<Dtype>(pt_a_h - roi_ctr_h) * aff_mat[1][1]) + roi_ctr_h;
      int r_pt_b_w = round(static_cast<Dtype>(pt_b_w - roi_ctr_w) * aff_mat[0][0] +
                           static_cast<Dtype>(pt_b_h - roi_ctr_h) * aff_mat[0][1]) + roi_ctr_w;
      int r_pt_b_h = round(static_cast<Dtype>(pt_b_w - roi_ctr_w) * aff_mat[1][0] +
                           static_cast<Dtype>(pt_b_h - roi_ctr_h) * aff_mat[1][1]) + roi_ctr_h;
      int r_pt_c_w = round(static_cast<Dtype>(pt_c_w - roi_ctr_w) * aff_mat[0][0] +
                           static_cast<Dtype>(pt_c_h - roi_ctr_h) * aff_mat[0][1]) + roi_ctr_w;
      int r_pt_c_h = round(static_cast<Dtype>(pt_c_w - roi_ctr_w) * aff_mat[1][0] +
                           static_cast<Dtype>(pt_c_h - roi_ctr_h) * aff_mat[1][1]) + roi_ctr_h;
      int r_pt_d_w = round(static_cast<Dtype>(pt_d_w - roi_ctr_w) * aff_mat[0][0] +
                           static_cast<Dtype>(pt_d_h - roi_ctr_h) * aff_mat[0][1]) + roi_ctr_w;
      int r_pt_d_h = round(static_cast<Dtype>(pt_d_w - roi_ctr_w) * aff_mat[1][0] +
                           static_cast<Dtype>(pt_d_h - roi_ctr_h) * aff_mat[1][1]) + roi_ctr_h;
      Dtype aa = (r_pt_b_w - r_pt_a_w) * (h - r_pt_a_h) - (r_pt_b_h - r_pt_a_h) * (w - r_pt_a_w);
      Dtype bb = (r_pt_c_w - r_pt_b_w) * (h - r_pt_b_h) - (r_pt_c_h - r_pt_b_h) * (w - r_pt_b_w);
      Dtype cc = (r_pt_d_w - r_pt_c_w) * (h - r_pt_c_h) - (r_pt_d_h - r_pt_c_h) * (w - r_pt_c_w);
      Dtype dd = (r_pt_a_w - r_pt_d_w) * (h - r_pt_d_h) - (r_pt_a_h - r_pt_d_h) * (w - r_pt_d_w);

      // Skip if Rotate ROI doesn't include (h, w)
      const bool in_roi = ((aa > Dtype(0.) && bb > Dtype(0.) && cc > Dtype(0.) && dd > Dtype(0.)) ||
                           (aa < Dtype(0.) && bb < Dtype(0.) && cc < Dtype(0.) && dd < Dtype(0.)));
      if (!in_roi) {
        continue;
      }

      int offset = (roi_n * channels + c) * pooled_height * pooled_width;
      const Dtype* offset_top_diff = top_diff + offset;
      const int* offset_argmax_data = argmax_data + offset;

      // Compute feasible set of pooled units that could have pooled
      // this bottom unit

      // Force malformed ROIs to be 1x1
      Dtype bin_size_h = static_cast<Dtype>(roi_height) / static_cast<Dtype>(pooled_height);
      Dtype bin_size_w = static_cast<Dtype>(roi_width) / static_cast<Dtype>(pooled_width);

      // invert rotate (w, h) to align box
      Dtype inv_aff_mat[2][2];
      inv_aff_mat[0][0] = static_cast<Dtype>(cos(-roi_theta));
      inv_aff_mat[0][1] = static_cast<Dtype>(sin(-roi_theta));
      inv_aff_mat[1][0] = static_cast<Dtype>(-sin(-roi_theta));
      inv_aff_mat[1][1] = static_cast<Dtype>(cos(-roi_theta));
      int inv_w = round(static_cast<Dtype>(w - roi_ctr_w) * inv_aff_mat[0][0] + 
                        static_cast<Dtype>(h - roi_ctr_h) * inv_aff_mat[0][1]) + roi_ctr_w;
      int inv_h = round(static_cast<Dtype>(w - roi_ctr_w) * inv_aff_mat[1][0] + 
                        static_cast<Dtype>(h - roi_ctr_h) * inv_aff_mat[1][1]) + roi_ctr_h;

      int phstart = floor(static_cast<Dtype>(inv_h - roi_start_h) / bin_size_h);
      int phend = ceil(static_cast<Dtype>(inv_h - roi_start_h + 1) / bin_size_h);
      int pwstart = floor(static_cast<Dtype>(inv_w - roi_start_w) / bin_size_w);
      int pwend = ceil(static_cast<Dtype>(inv_w - roi_start_w + 1) / bin_size_w);

      phstart = min(max(phstart, 0), pooled_height);
      phend = min(max(phend, 0), pooled_height);
      pwstart = min(max(pwstart, 0), pooled_width);
      pwend = min(max(pwend, 0), pooled_width);

      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (offset_argmax_data[ph * pooled_width + pw] == (h * width + w)) {
            gradient += offset_top_diff[ph * pooled_width + pw];
          }
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
void RotateROIPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  const int* argmax_data = max_idx_.gpu_data();
  // NOLINT_NEXT_LINE(whitespace/operators)
  RotateROIPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, argmax_data, top[0]->num(), spatial_scale_, channels_,
      height_, width_, pooled_height_, pooled_width_, bottom_diff, bottom_rois);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(RotateROIPoolingLayer);

}  // namespace caffe
