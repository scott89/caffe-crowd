#include <vector>
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/common_layers.hpp"

namespace caffe {

template<typename Dtype>
void PerspectiveLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
       const vector<Blob<Dtype>*>& top){
  CHECK_EQ(bottom.size(), 3) << "Currently support exact 3 bottoms: "
    << "1. slope, 2. intercept, 3. reference blob.";
  CHECK_EQ(bottom[0]->count(1), 1)
    << "First bottom blob is slope, which should be a scalar for each sample.";
  CHECK_EQ(bottom[1]->count(1), 1)
    << "First bottom blob is intercept, which should be a scalar for each sample.";
  num_ = bottom[0]->num();
  height_ = bottom[2]->height();
  width_ = bottom[2]->width();
  channels_ = 1;

  // Initialize multipliers
  slope_multiplier_.Reshape(1, 1, height_, width_);
  intercept_multiplier_.Reshape(1, 1, height_, width_);
  Blob<Dtype> slope_row_fac, slope_col_fac, intercept_row_fac, intercept_col_fac;
  slope_row_fac.Reshape(1, 1, height_, 1);
  intercept_row_fac.Reshape(1, 1, height_, 1);
  for (int i = 0; i < height_; i++) {
    slope_row_fac.mutable_cpu_data()[i] = i;
  }
  caffe_set<Dtype>(height_, (Dtype)1.,
                   intercept_row_fac.mutable_cpu_data());

  slope_col_fac.Reshape(1, 1, 1, width_);
  intercept_col_fac.Reshape(1, 1, 1, width_);
  caffe_set(width_, (Dtype)1., slope_col_fac.mutable_cpu_data());
  caffe_set(width_, (Dtype)1., intercept_col_fac.mutable_cpu_data());

  Dtype slope_mult = this->layer_param_.perspective_param().slope_mult();
  Dtype intercept_mult = this->layer_param_.perspective_param().intercept_mult();
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, height_, width_, 1,
      slope_mult, slope_row_fac.cpu_data(), slope_col_fac.cpu_data(),
      (Dtype)0., slope_multiplier_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, height_, width_, 1,
      intercept_mult, intercept_row_fac.cpu_data(), intercept_col_fac.cpu_data(),
      (Dtype)0., intercept_multiplier_.mutable_cpu_data());
}

template<typename Dtype>
void PerspectiveLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
             const vector<Blob<Dtype>*>& top){
  top[0]->Reshape(num_, 1, height_, width_);
}

template<typename Dtype>
void PerspectiveLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
       const vector<Blob<Dtype>*>& top){
  for (int n = 0; n < num_; n++) {
    const Dtype slope = bottom[0]->cpu_data()[bottom[0]->offset(n)];
    const Dtype intercept = bottom[1]->cpu_data()[bottom[1]->offset(n)];
    Dtype* cur_map = top[0]->mutable_cpu_data() + top[0]->offset(n);
    caffe_cpu_axpby<Dtype>(height_ * width_,
                           slope, slope_multiplier_.cpu_data(),
                           Dtype(0.), cur_map);
    caffe_cpu_axpby<Dtype>(height_ * width_,
                           intercept, intercept_multiplier_.cpu_data(),
                           Dtype(1.), cur_map);
  }
}

template<typename Dtype>
void PerspectiveLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
  const Dtype* top_diff = top[0]->cpu_diff();
  for (int n = 0; n < num_; n++) {
    const Dtype* slope_multipliers = slope_multiplier_.cpu_data();
    const Dtype* intercept_multipliers = intercept_multiplier_.cpu_data();
    Dtype slope_diff = caffe_cpu_dot<Dtype>(height_ * width_,
        top_diff + top[0]->offset(n), slope_multipliers);
    Dtype intercept_diff = caffe_cpu_dot<Dtype>(height_ * width_,
        top_diff + top[0]->offset(n), intercept_multipliers);

    bottom[0]->mutable_cpu_diff()[n] = slope_diff;
    bottom[1]->mutable_cpu_diff()[n] = intercept_diff;
  }
}



#ifdef CPU_ONLY
STUB_GPU(PerspectiveLayer);
#endif

INSTANTIATE_CLASS(PerspectiveLayer);
REGISTER_LAYER_CLASS(Perspective);

}   // namespace caffe