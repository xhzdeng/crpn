#include <vector>

#include "caffe/layers/euclidean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  has_weights_ = (bottom.size() >= 3);
  if (has_weights_) {
    CHECK_EQ(bottom.size(), 4) << "If weights are used, must specify both "
      "inside and outside weights";
  }
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  if (has_weights_) {
    CHECK_EQ(bottom[0]->channels(), bottom[2]->channels());
    CHECK_EQ(bottom[0]->height(), bottom[2]->height());
    CHECK_EQ(bottom[0]->width(), bottom[2]->width());
    CHECK_EQ(bottom[0]->channels(), bottom[3]->channels());
    CHECK_EQ(bottom[0]->height(), bottom[3]->height());
    CHECK_EQ(bottom[0]->width(), bottom[3]->width());
  }
  diff_.ReshapeLike(*bottom[0]);
  // vector of ones used to sum
  ones_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  for (int i = 0; i < bottom[0]->count(); ++i) {
    ones_.mutable_cpu_data()[i] = Dtype(1);
  }
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
  // int count = bottom[0]->count();
  // caffe_sub(
  //     count,
  //     bottom[0]->cpu_data(),
  //     bottom[1]->cpu_data(),
  //     diff_.mutable_cpu_data());
  // Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  // Dtype loss = dot / bottom[0]->num() / Dtype(2);
  // top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
  // for (int i = 0; i < 2; ++i) {
  //   if (propagate_down[i]) {
  //     const Dtype sign = (i == 0) ? 1 : -1;
  //     const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
  //     caffe_cpu_axpby(
  //         bottom[i]->count(),              // count
  //         alpha,                              // alpha
  //         diff_.cpu_data(),                   // a
  //         Dtype(0),                           // beta
  //         bottom[i]->mutable_cpu_diff());  // b
  //   }
  // }
}

#ifdef CPU_ONLY
STUB_GPU(EuclideanLossLayer);
#endif

INSTANTIATE_CLASS(EuclideanLossLayer);
REGISTER_LAYER_CLASS(EuclideanLoss);

}  // namespace caffe
