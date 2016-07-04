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
