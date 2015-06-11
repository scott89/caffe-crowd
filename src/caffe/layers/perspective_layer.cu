#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template<typename Dtype>
void PerspectiveLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
       const vector<Blob<Dtype>*>& top){
  for (int n = 0; n < num_; n++) {
    const Dtype slope = bottom[0]->cpu_data()[n];
    const Dtype intercept = bottom[1]->cpu_data()[n];
    Dtype* cur_map = top[0]->mutable_gpu_data() + top[0]->offset(n);
    caffe_gpu_axpby<Dtype>(height_ * width_, slope,
        slope_multiplier_.gpu_data(), Dtype(0.), cur_map);
    caffe_gpu_axpby<Dtype>(height_ * width_, intercept,
        intercept_multiplier_.gpu_data(), Dtype(1.), cur_map);
  }
}

template<typename Dtype>
void PerspectiveLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
  for (int n = 0; n < num_; n++) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* slope_multipliers = slope_multiplier_.gpu_data();
    const Dtype* intercept_multipliers = intercept_multiplier_.gpu_data();
    Dtype slope_diff, intercept_diff;
    caffe_gpu_dot<Dtype>(height_ * width_, top_diff + top[0]->offset(n),
        slope_multipliers, &slope_diff);
    caffe_gpu_dot<Dtype>(height_ * width_, top_diff + top[0]->offset(n),
        intercept_multipliers, &intercept_diff);
    bottom[0]->mutable_cpu_diff()[n] = slope_diff;
    bottom[1]->mutable_cpu_diff()[n] = intercept_diff;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(PerspectiveLayer);
}  // namespace caffe