#ifndef CAFFE_NEURON_LAYER_HPP_
#define CAFFE_NEURON_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
/*
������data�󣬾�Ҫ�����ˣ����糣����sigmoid��tanh�ȵȣ���Щ������������������
neuron_layers.hpp�������NeuronLayer�������ֻ�������ļ��㣬�����ȷ��������
��ExactNumBottomBlobs()��ExactNumTopBlobs()���ǳ���1,������һ��blob�����һ��blob��

caffe�ж������6�ֳ��õļ������ReLu��Sigmod��Tanh��Absval��Power��BNll;������Ҫ���������ֽ���˵����
1. 6�ּ�����Ķ���
2. caffe�ж�6�ּ������ķ�װ
3. caffe�����ʹ��6�ּ����������򵥣�

1. 6�ּ�����Ķ���
  1.1 ReLU / Rectified-Linear and Leaky-ReLU
     ReLU��Ŀǰʹ�����ļ��������Ҫ��Ϊ���������죬�����ܱ���ͬ��Ч����
��׼��ReLU����Ϊmax(x, 0)����x>0ʱ�����x; ��x<=0ʱ�����0
         f(x)=max(x,0)
  1.2 Sigmoid
  1.3 TanH / Hyperbolic Tangent
  1.4 Absolute Value
     f(x)=Abs(x)
  1.5 Power
     f(x)= (shift + scale * x) ^ power
  1.6 BNLL binomial normal log likelihood�ļ��
     f(x)=log(1 + exp(x))

2. caffe�ж�6�ּ������ķ�װ������Դ��Ľ���������һƪ�����У�

3. caffe�����ʹ��6�ּ����

   ��Ҫ�Ƕ��ڡ�type�����޸ģ�Ȼ���в������ٶ�������Ϳ�����




Activiation / Neuron Layers����������Ԫ��
 

ͨ���£�����layer����element-wise����������һ��bottom blob������һ��ͬ����С��blob���������layer�����У����Ǻ��������������С����Ϊ��������ͬ�ģ�����n * c * h * w��

 

ReLU / Rectified-Linear and Leaky-ReLU:

layer���ͣ�ReLU
CPUʵ�֣�./src/caffe/layers/relu_layer.cpp
CUDA GPUʵ�֣�./src/caffe/layers/relu_layer.cu
����(ReLUParameter relu_param)
��ѡ��
negative_slope [default 0]: ָ���Ƿ�ʹ��б��ֵ���渺�����֣����ǽ���������ֱ������Ϊ0.
���ӣ�
1
2
3
4
5
6
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "conv1"
  top: "conv1"
}
����һ������ֵx��ReLU����x > 0ʱ���x�� x < 0ʱ���negative_slope * x����negative_slope����û������ʱ���ȼ��ڱ�׼ReLU����(max(x, 0))����֧��ԭλ���㣬��ζ��bottom��top blob��ַͬ�ģ��������ڴ����ġ�

 

Sigmoid:

layer���ͣ�Sigmoid
CPUʵ�֣�./src/caffe/layers/sigmoid_layer.cpp
CUDA GPUʵ�֣�./src/caffe/layers/sigmoid_layer.cu
���ӣ�
layer {
  name: "encode1neuron"
  bottom: "encode1"
  top: "encode1neuron"
  type: "Sigmoid"
}
 

TanH / Hyperbolic Tangent
layer���ͣ�TanH
CPUʵ�֣�./src/caffe/layers/tanh_layer.cpp
CUDA GPUʵ�֣�./src/caffe/layers/tanh_layer.cu
���ӣ�
layer {
  name: "layer"
  bottom: "in"
  top: "out"
  type: "TanH"
}
 

Absolute Value
layer���ͣ�AbsVal
CPUʵ�֣�./src/caffe/layers/absval_layer.cpp
CUDA GPUʵ�֣�./src/caffe/layers/absval_layer.cu
���ӣ�
layer {
  name: "layer"
  bottom: "in"
  top: "out"
  type: "AbsVal"
}
 

Power
layer���ͣ�Power
CPUʵ�֣�./src/caffe/layers/power_layer.cpp
CUDA GPUʵ�֣�./src/caffe/layers/power_layer.cu
����(PowerParameter power_param)
��ѡ��
power [default 1]
scale [default 1]
shift [default 0]
���ӣ�
���ƴ���
layer {
  name: "layer"
  bottom: "in"
  top: "out"
  type: "Power"
  power_param {
    power: 1
    scale: 1
    shift: 0
  }
}
���ƴ���
power���������Ϊxʱ�ģ����Ϊ(shift + scale * x)^power��

 

BNLL (Binomial Normal Log Likelihood) ����ʽ��׼������Ȼ
layer���ͣ�BNLL
CPUʵ�֣�./src/caffe/layers/bnll_layer.cpp
CUDA GPUʵ�֣�./src/caffe/layers/bnll_layer.cu
���ӣ�
layer {
  name: "layer"
  bottom: "in"
  top: "out"
  type: BNLL
}
BNLL layer��������x�����Ϊlog(1 + exp(x))��   
*/

namespace caffe {

/**
 * @brief An interface for layers that take one blob as input (@f$ x @f$)
 *        and produce one equally-sized blob as output (@f$ y @f$), where
 *        each element of the output depends only on the corresponding input
 *        element.
 */
template <typename Dtype>
class NeuronLayer : public Layer<Dtype> {
 public:
  explicit NeuronLayer(const LayerParameter& param)
     : Layer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
};

}  // namespace caffe

#endif  // CAFFE_NEURON_LAYER_HPP_
