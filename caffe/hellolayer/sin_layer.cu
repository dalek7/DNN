// Sin neuron activation function layer.
// Adapted from TanH layer which was adapted from the ReLU layer code written by Yangqing Jia

#include <vector>

#include "caffe/layers/sin_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SinForward(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = sin(in[index]);
  }
}

template <typename Dtype>
void SinLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  SinForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void SinForward(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = sin(in[index]);
  }
}

template <typename Dtype>
void SinLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  SinForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(SinLayer);


}  // namespace caffe


