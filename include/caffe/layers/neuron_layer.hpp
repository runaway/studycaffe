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
