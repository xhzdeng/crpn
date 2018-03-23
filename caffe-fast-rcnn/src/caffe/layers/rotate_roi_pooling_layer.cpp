// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include <cfloat>

#include "caffe/fast_rcnn_layers.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
void RotateROIPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  RotateROIPoolingParameter rotate_roi_pool_param = this->layer_param_.rotate_roi_pooling_param();
  CHECK_GT(rotate_roi_pool_param.pooled_h(), 0)
      << "pooled_h must be > 0";
  CHECK_GT(rotate_roi_pool_param.pooled_w(), 0)
      << "pooled_w must be > 0";
  pooled_height_ = rotate_roi_pool_param.pooled_h();
  pooled_width_ = rotate_roi_pool_param.pooled_w();
  spatial_scale_ = rotate_roi_pool_param.spatial_scale();
  LOG(INFO) << "Spatial scale: " << spatial_scale_;
}

template <typename Dtype>
void RotateROIPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  top[0]->Reshape(bottom[1]->num(), channels_, pooled_height_,
      pooled_width_);
  max_idx_.Reshape(bottom[1]->num(), channels_, pooled_height_,
      pooled_width_);
}

template <typename Dtype>
void RotateROIPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_rois = bottom[1]->cpu_data();
  // Number of ROIs
  int num_rois = bottom[1]->num();
  int batch_size = bottom[0]->num();
  int top_count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top_count, Dtype(-FLT_MAX), top_data);
  int* argmax_data = max_idx_.mutable_cpu_data();
  caffe_set(top_count, -1, argmax_data);

  // For each ROI R = [batch_index x1 y1 x2 y2 theta]: max pool over R
  for (int n = 0; n < num_rois; ++n) {
    int roi_batch_ind = bottom_rois[0];
    int roi_start_w = round(bottom_rois[1] * spatial_scale_);
    int roi_start_h = round(bottom_rois[2] * spatial_scale_);
    int roi_end_w = round(bottom_rois[3] * spatial_scale_);
    int roi_end_h = round(bottom_rois[4] * spatial_scale_);
    float roi_theta = bottom_rois[5] / 180.f * 3.1415926;

    // debug
    // std::cout << "roi_start_w: "<< roi_start_w << std::endl;
    // std::cout << "roi_start_h: " << roi_start_h << std::endl; 
    // std::cout << "roi_end_w: "<< roi_end_w << std::endl;
    // std::cout << "roi_end_h: " << roi_end_h << std::endl; 
    // std::cout << "roi_theta: "<< roi_theta << std::endl;   

    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    int roi_ctr_w = roi_start_w + roi_width / 2;
    int roi_ctr_h = roi_start_h + roi_height / 2;

    // get affine matrix
    Dtype affine[2][2];
    affine[0][0] = cos(roi_theta);
    affine[0][1] = sin(roi_theta);
    affine[1][0] = -sin(roi_theta);
    affine[1][1] = cos(roi_theta);   

    CHECK_GE(roi_batch_ind, 0);
    CHECK_LT(roi_batch_ind, batch_size);

    const Dtype bin_size_h = static_cast<Dtype>(roi_height) / static_cast<Dtype>(pooled_height_);
    const Dtype bin_size_w = static_cast<Dtype>(roi_width) / static_cast<Dtype>(pooled_width_);

    const Dtype* batch_data = bottom_data + bottom[0]->offset(roi_batch_ind);

    for (int c = 0; c < channels_; ++c) {
      for (int ph = 0; ph < pooled_height_; ++ph) {
        for (int pw = 0; pw < pooled_width_; ++pw) {
          // Compute pooling region for this output unit:
          // start (included) = floor(ph * roi_height / pooled_height_)
          // end (excluded) = ceil((ph + 1) * roi_height / pooled_height_)
          int hstart = static_cast<int>(floor(static_cast<Dtype>(ph) * bin_size_h));
          int wstart = static_cast<int>(floor(static_cast<Dtype>(pw) * bin_size_w));
          int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1) * bin_size_h));
          int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1) * bin_size_w));

          hstart = min(max(hstart + roi_start_h, 0), height_);
          hend = min(max(hend + roi_start_h, 0), height_);
          wstart = min(max(wstart + roi_start_w, 0), width_);
          wend = min(max(wend + roi_start_w, 0), width_);

          bool is_empty = (hend <= hstart) || (wend <= wstart);

          const int pool_index = ph * pooled_width_ + pw;
          if (is_empty) {
            top_data[pool_index] = 0;
            argmax_data[pool_index] = -1;
          }

          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              // rotate
              int r_w = round(static_cast<Dtype>(w - roi_ctr_w) * affine[0][0] + 
                              static_cast<Dtype>(h - roi_ctr_h) * affine[0][1]) + roi_ctr_w;
              int r_h = round(static_cast<Dtype>(w - roi_ctr_w) * affine[1][0] + 
                              static_cast<Dtype>(h - roi_ctr_h) * affine[1][1]) + roi_ctr_h;
              // filter illegal point
              if ((r_w < 0) || (r_h < 0) || (r_w > width_ - 1) || (r_h > height_ - 1))
                continue;
              // const int index = h * width_ + w;
              const int index = r_h * width_ + r_w;
              if (batch_data[index] > top_data[pool_index]) {
                top_data[pool_index] = batch_data[index];
                argmax_data[pool_index] = index;
              }
            }
          }
          // debug
          // std::cout << "Pool Index: " << pool_index << std::endl;
          // std::cout << "Map Index: " << argmax_data[pool_index] << std::endl;
          // std::cout << "Value: "<< top_data[pool_index] << std::endl;
        }
      }
      // Increment all data pointers by one channel
      batch_data += bottom[0]->offset(0, 1);
      top_data += top[0]->offset(0, 1);
      argmax_data += max_idx_.offset(0, 1);
    }
    // Increment ROI data pointer
    bottom_rois += bottom[1]->offset(1);
  }
}

template <typename Dtype>
void RotateROIPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(RotateROIPoolingLayer);
#endif

INSTANTIATE_CLASS(RotateROIPoolingLayer);
REGISTER_LAYER_CLASS(RotateROIPooling);

}  // namespace caffe
