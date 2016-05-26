#include <algorithm>
#include <vector>

#include "caffe/layers/relu_layer.hpp"
/*
Relu 函数是一个比较流行的激活函数，已经逐渐取代sigmoid函数。它不会随着输入 z 的逐渐增加而趋于饱和。
多层的神经网络如果用sigmoid或tanh激活函数也不做pre-training的话会因为 gradient vanishing problem 而会无法收敛。使用ReLU则这没有这个问题。

预训练的用处：规则化，防止过拟合；压缩数据，去除冗余；强化特征，减小误差；加快收敛速度。

标准的sigmoid输出不具备稀疏性，需要用一些惩罚因子来训练出一大堆接近0的冗余数据来，从而产生稀疏数据，例如L1、L2作惩罚因子。因此需要进行无监督的预训练。

而ReLU是线性修正，公式为：g(x) = max(0, x)，是purelin的折线版。它的作用是如果计算出的值小于0，就让它等于0，否则保持原来的值不变。

这是一种简单粗暴地强制某些数据为0的方法，然而经实践证明，训练后的网络完全具备适度的稀疏性。而且训练后的可视化效果和传统方式预训练出的效果很相似，这也说明了ReLU具备引导适度稀疏的能力。

而后续又出现了relu的修改版prelu，这里不做介绍。
*/
namespace caffe {

template <typename Dtype>
void ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  for (int i = 0; i < count; ++i) {
    top_data[i] = std::max(bottom_data[i], Dtype(0))
        + negative_slope * std::min(bottom_data[i], Dtype(0));
  }
}

template <typename Dtype>
void ReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + negative_slope * (bottom_data[i] <= 0));
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ReLULayer);
#endif

INSTANTIATE_CLASS(ReLULayer);

}  // namespace caffe
