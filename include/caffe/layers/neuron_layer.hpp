#ifndef CAFFE_NEURON_LAYER_HPP_
#define CAFFE_NEURON_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
/*
输入了data后，就要计算了，比如常见的sigmoid、tanh等等，这些都计算操作被抽象成了
neuron_layers.hpp里面的类NeuronLayer，这个层只负责具体的计算，因此明确定义了输
入ExactNumBottomBlobs()和ExactNumTopBlobs()都是常量1,即输入一个blob，输出一个blob。

caffe中定义好了6种常用的激活函数：ReLu、Sigmod、Tanh、Absval、Power、BNll;下面主要从两个部分进行说明。
1. 6种激活函数的定义
2. caffe中对6种激活函数类的封装
3. caffe中如何使用6种激活函数（极其简单）

1. 6种激活函数的定义
  1.1 ReLU / Rectified-Linear and Leaky-ReLU
     ReLU是目前使用最多的激活函数，主要因为其收敛更快，并且能保持同样效果。
标准的ReLU函数为max(x, 0)，当x>0时，输出x; 当x<=0时，输出0
         f(x)=max(x,0)
  1.2 Sigmoid
  1.3 TanH / Hyperbolic Tangent
  1.4 Absolute Value
     f(x)=Abs(x)
  1.5 Power
     f(x)= (shift + scale * x) ^ power
  1.6 BNLL binomial normal log likelihood的简称
     f(x)=log(1 + exp(x))

2. caffe中对6种激活函数类的封装（对于源码的解析放在下一篇博文中）

3. caffe中如何使用6种激活函数

   主要是对于“type”的修改，然后有参数的再定义参数就可以了




Activiation / Neuron Layers：激励或神经元层
 

通常下，这类layer都是element-wise操作，输入一个bottom blob，产生一个同样大小的blob。在下面的layer介绍中，我们忽略了输入输出大小，因为它们是相同的，都是n * c * h * w。

 

ReLU / Rectified-Linear and Leaky-ReLU:

layer类型：ReLU
CPU实现：./src/caffe/layers/relu_layer.cpp
CUDA GPU实现：./src/caffe/layers/relu_layer.cu
参数(ReLUParameter relu_param)
可选的
negative_slope [default 0]: 指定是否使用斜坡值代替负数部分，还是将负数部分直接设置为0.
例子：
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
给定一个输入值x，ReLU层在x > 0时输出x， x < 0时输出negative_slope * x。当negative_slope参数没有设置时，等价于标准ReLU函数(max(x, 0))。它支持原位运算，意味着bottom和top blob是同址的，减少了内存消耗。

 

Sigmoid:

layer类型：Sigmoid
CPU实现：./src/caffe/layers/sigmoid_layer.cpp
CUDA GPU实现：./src/caffe/layers/sigmoid_layer.cu
例子：
layer {
  name: "encode1neuron"
  bottom: "encode1"
  top: "encode1neuron"
  type: "Sigmoid"
}
 

TanH / Hyperbolic Tangent
layer类型：TanH
CPU实现：./src/caffe/layers/tanh_layer.cpp
CUDA GPU实现：./src/caffe/layers/tanh_layer.cu
例子：
layer {
  name: "layer"
  bottom: "in"
  top: "out"
  type: "TanH"
}
 

Absolute Value
layer类型：AbsVal
CPU实现：./src/caffe/layers/absval_layer.cpp
CUDA GPU实现：./src/caffe/layers/absval_layer.cu
例子：
layer {
  name: "layer"
  bottom: "in"
  top: "out"
  type: "AbsVal"
}
 

Power
layer类型：Power
CPU实现：./src/caffe/layers/power_layer.cpp
CUDA GPU实现：./src/caffe/layers/power_layer.cu
参数(PowerParameter power_param)
可选的
power [default 1]
scale [default 1]
shift [default 0]
例子：
复制代码
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
复制代码
power层计算输入为x时的，输出为(shift + scale * x)^power。

 

BNLL (Binomial Normal Log Likelihood) 二项式标准对数似然
layer类型：BNLL
CPU实现：./src/caffe/layers/bnll_layer.cpp
CUDA GPU实现：./src/caffe/layers/bnll_layer.cu
例子：
layer {
  name: "layer"
  bottom: "in"
  top: "out"
  type: BNLL
}
BNLL layer计算输入x的输出为log(1 + exp(x))。   
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
