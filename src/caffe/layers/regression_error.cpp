#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void RegressionErrorLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  //top_k_ = this->layer_param_.accuracy_param().top_k();
}

template <typename Dtype>
void RegressionErrorLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  //Layer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  CHECK_EQ(bottom[0]->channels() * bottom[0]->height() * bottom[0]->width(), bottom[1]->channels()*bottom[1]->height()*bottom[1]->width());
  //CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  //CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  bottom[1]->Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  (*top)[0]->Reshape(1, 1, 1, 1);
///////////////////////////////////////////////////
//  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
//      << "The data and label should have the same number.";
  //CHECK_LE(top_k_, bottom[0]->count() / bottom[0]->num())
  //    << "top_k must be less than or equal to the number of classes.";
//  CHECK_EQ(bottom[1]->channels(), 1);
//  CHECK_EQ(bottom[1]->height(), 1);
//  CHECK_EQ(bottom[1]->width(), 1);
//  (*top)[0]->Reshape(1, 1, 1, 1);
}

template <typename Dtype>
void RegressionErrorLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {

  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  //Dtype loss = dot / bottom[0]->num() / Dtype(2);
  Dtype loss = dot / bottom[0]->num()/bottom[0]->height()/ bottom[0]->width();// / Dtype(2);
  (*top)[0]->mutable_cpu_data()[0] = loss;
///////////////////////////////////////////////////
}

INSTANTIATE_CLASS(RegressionErrorLayer);

}  // namespace caffe
