#ifndef CAFFE_SIN_LAYER_HPP_
#define CAFFE_SIN_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

template <typename Dtype>
class SinLayer : public NeuronLayer<Dtype> {
public:
  explicit SinLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}

  virtual inline const char* type() const { return "Sin"; }


protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
#if 0
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
#endif
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

#if 0
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

#endif
};

}  // namespace caffe

#endif  // CAFFE_SIN_LAYER_HPP_

