#include <vector>

#include "caffe/layers/loss_layer.hpp"

/*
损失函数，一般由两项组成，一项是loss term,另外一项是regularization term。

J=L+R

先说损失项loss，再说regularization项。

1. 分对得分1，分错得分0.gold standard

2. hinge loss(for softmargin svm),J=1/2||w||^2 + sum(max(0,1-yf(w,x)))

3. log los, cross entropy loss function in logistic regression model.
   J=lamda||w||^2+sum(log(1+e(-yf(wx))))

4. squared loss, in linear regression. loss=(y-f(w,x))^2

5. exponential loss in boosting. J=lambda*R+exp(-yf(w,x))

再说regularization项，

一般用的多的是R2=1/2||w||^2,R1=sum(|w|)。R1和R2是凸的，同时R1会使得损失函数更加
具有sparse，而R2则会更加光滑些。具体可以参见下图：

caffe的损失函数，目前已经囊括了所有可以用的了吧，损失函数由最后一层分类器决定，
同时有时会加入regularization,在BP过程中，使得误差传递得以良好运行。 

contrastive_loss，对应contrastive_loss_layer，我看了看代码，这个应该是输入是一
对用来做验证的数据，比如两张人脸图，可能是同一个人的（正样本），也可能是不同个
人（负样本）。在caffe的examples中，siamese这个例子中，用的损失函数是该类型的。
该损失函数具体数学表达形式可以参考lecun的文章Dimensionality Reduction by 
Learning an Invariant Mapping, Raia Hadsell, Sumit Chopra, Yann LeCun, cvpr 2006. 

euclidean_loss，对应euclidean_loss_layer,该损失函数就是l=(y-f(wx))^2，是线性回
归常用的损失函数。

hinge_loss，对应hinge_loss_layer，该损失函数就是\ell(y) = \max(0, 1-t \cdot y)。
主要用在SVM分类器中。

infogain_loss，对应infogain_loss_layer，损失函数表达式没找到，只知道这是在文本
处理中用到的损失函数。

multinomial_logistic_loss，对应multinomial_logistic_loss_layer，

sigmoid_cross_entropy，对应sigmoid_cross_entropy_loss_layer,也就是logistic 
regression使用的损失函数。

softmax_loss,对应softmax_loss_layer，损失函数等可以见UFLDL中关于softmax章节。在
caffe中多类分类问题，损失函数就是softmax_loss，比如imagenet, mnist等。
softmax_loss是sigmoid的多类问题。但是，我就没明白，multinomial_logistic_loss和
这个有什么区别，看代码，输入有点差别，softmax的输入是probability,而multinomial
好像不要求是probability，但是还是没明白，如果只是这样，岂不是一样啊？

这里详细说明了两者之间的差异，并且有详细的测试结果，非常赞。简单理解，multinomial
是将loss分成两个层进行，而softmax则是合在一起了。或者说，multinomial loss是按部
就班的计算反向梯度，而softmax则是把两个步骤直接合并为一个步骤进行了，减少了中间
的精度损失等 ，从计算稳定性讲，softmax更好，multinomial是标准做法，softmax则是
一种优化吧。




loss驱动了学习过程，它比较输出和目标之间的差异，并为之设置代价去最小化。loss本身被前向传播计算，关于loss的梯度被后向传播计算。
 

Softmax：type: SoftmaxWithLoss

softmax损失层计算输入的softmax的多项式logistic loss。它概念上等同于一个softmax layer，后面连接一个多项式logistic loss layer，但是softmax loss layer提供一个数值更稳定的梯度。

 

Sum-of-Squares / Euclidean: type: EuclideanLoss

Euclidean损失层计算两个输入的差的平方和：

 

Hinge / Margin: 铰链损失或边缘损失

layer类型：HingeLoss
CPU实现：./src/caffe/layers/hinge_loss_layer.cpp
CUDA GPU实现：目前尚无GPU实现
参数(HingeLossParameter hinge_loss_param)
可选的
norm [default L1]: 使用范数，目前包括 L1, L2两种选择。
输入：
n * c * h * w Predictions
n * 1 * 1 * 1 Labels
输出：1 * 1 * 1 * 1 所得损失
例子：
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
# L1 Norm
layer {
  name: "loss"
  type: "HingeLoss"
  bottom: "pred"
  bottom: "label"
}
 
# L2 Norm
layer {
  name: "loss"
  type: "HingeLoss"
  bottom: "pred"
  bottom: "label"
  top: "loss"
  hinge_loss_param {
    norm: L2
  }
}
 

Sigmoid Cross-Entropy: type: SigmoidCrossEntropyLoss交叉熵损失，用于多标签分类

 

Infogain: type: InfogainLoss信息增益损失

 

Accuracy and Top-K: 准确性对输出进行评分，计算输出与目标之间的差异，它实际上不是一个loss，没有后向传播阶段。
*/

namespace caffe {

template <typename Dtype>
void LossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // LossLayers have a non-zero (1) loss by default.
  if (this->layer_param_.loss_weight_size() == 0) {
    this->layer_param_.add_loss_weight(Dtype(1));
  }
}

template <typename Dtype>
void LossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.

  // 输出top是一个标量。
  top[0]->Reshape(loss_shape);
}

INSTANTIATE_CLASS(LossLayer);

}  // namespace caffe
