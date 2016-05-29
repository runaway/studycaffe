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
    // 输入大于零斜率为1，小于0斜率为negative_slope。
    top_data[i] = std::max(bottom_data[i], Dtype(0))
        + negative_slope * std::min(bottom_data[i], Dtype(0));
  }
}
/*
关于Backward_cpu()作几点说明： 
propagate_down：在caffe.proto里有一段说明：Specifies on which bottoms the backpropagation should be skipped.The size must be either 0 or equal to the number of bottoms. propagate_down是与计算关于bottom的梯度相关的。大家在通常的理解上，求梯度都是相对于参数weights而言的，但是，在caffe里为什么还有求“bottom的导数”一说呢？？？原因在于caffe的实现。公式（1）中，δ(l)其实就是损失函数关于当前层的输入（bottom）的偏导数，而这个 propagate_down则是计算这个δ(l)的控制条件，由公式就可以知道，这个δ(l)在caffe的BP实现中是非常重要。

caffe中BP的实现： caffe的模块化非常强，它将W与X的线性求和，激励函数以及pooling都分开了，也就是说caffe里的conv_layer只是一个线性求和运算，并没有激励运算，而且分别对应了caffe里的conv_layer, Relu_layer, 以及pooling_layer。与公式（3）对应，我们以Relu_layer为例来说明一下caffe的BP实现，公式中的δ(l+1)其实与top_diff对应，δ(l)其实与bottom_diff对应。而top_diff = top[0]->cpu_diff()，bottom_diff = bottom[0]->mutable_cpu_diff()，又因为Relu_layer并没有weight（卷积核），所以公式（3）化简为δ(l)=δ(l+1)?f′(z(l))， δ(l+1)来自上一层,f′(z(l))与代码中的for循环语句块对应。而对于conv_layer，公式（3）化简为δ(l)=(W(l))Tδ(l+1)，可以查看ConvolutionLayer::Backward_cpu()以验证。对应于公式（4），求的是关于weights卷积核的梯度，a(l)与当前卷积层的bottom相关。总之，基于caffe的高度模块性，其BP实现的梯度有关于bottom的，也有关于weights的。

公式（2）其实是公式（1）中J取EuclideanLoss的特例，(y?a(nl))就是top_diff。基于此，可以理解happynear大神所说的：CNN特征 = FP（输入图像）， diff = 原始特征 - CNN特征， loss = lossfunction（diff），新的grad=BP（loss）
*/
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
