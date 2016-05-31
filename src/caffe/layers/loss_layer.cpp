#include <vector>

#include "caffe/layers/loss_layer.hpp"

/*
��ʧ������һ����������ɣ�һ����loss term,����һ����regularization term��

J=L+R

��˵��ʧ��loss����˵regularization�

1. �ֶԵ÷�1���ִ�÷�0.gold standard

2. hinge loss(for softmargin svm),J=1/2||w||^2 + sum(max(0,1-yf(w,x)))

3. log los, cross entropy loss function in logistic regression model.J=lamda||w||^2+sum(log(1+e(-yf(wx))))

4. squared loss, in linear regression. loss=(y-f(w,x))^2

5. exponential loss in boosting. J=lambda*R+exp(-yf(w,x))

��˵regularization�

һ���õĶ����R2=1/2||w||^2,R1=sum(|w|)��R1��R2��͹�ģ�ͬʱR1��ʹ����ʧ�������Ӿ���sparse����R2�����ӹ⻬Щ��������Բμ���ͼ��


caffe����ʧ������Ŀǰ�Ѿ����������п����õ��˰ɣ���ʧ���������һ�������������ͬʱ��ʱ�����regularization,��BP�����У�ʹ�����ݵ����������С� 

contrastive_loss����Ӧcontrastive_loss_layer���ҿ��˿����룬���Ӧ����������һ����������֤�����ݣ�������������ͼ��������ͬһ���˵ģ�����������Ҳ�����ǲ�ͬ���ˣ�������������caffe��examples�У�siamese��������У��õ���ʧ�����Ǹ����͵ġ�����ʧ����������ѧ�����ʽ���Բο�lecun������Dimensionality Reduction by Learning an Invariant Mapping, Raia Hadsell, Sumit Chopra, Yann LeCun, cvpr 2006. 

euclidean_loss����Ӧeuclidean_loss_layer,����ʧ��������l=(y-f(wx))^2�������Իع鳣�õ���ʧ������

hinge_loss����Ӧhinge_loss_layer������ʧ��������\ell(y) = \max(0, 1-t \cdot y)����Ҫ����SVM�������С�

infogain_loss����Ӧinfogain_loss_layer����ʧ�������ʽû�ҵ���ֻ֪���������ı��������õ�����ʧ������

multinomial_logistic_loss����Ӧmultinomial_logistic_loss_layer��

sigmoid_cross_entropy����Ӧsigmoid_cross_entropy_loss_layer,Ҳ����logistic regressionʹ�õ���ʧ������

softmax_loss,��Ӧsoftmax_loss_layer����ʧ�����ȿ��Լ�UFLDL�й���softmax�½ڡ���caffe�ж���������⣬��ʧ��������softmax_loss������imagenet, mnist�ȡ�softmax_loss��sigmoid�Ķ������⡣���ǣ��Ҿ�û���ף�multinomial_logistic_loss�������ʲô���𣬿����룬�����е���softmax��������probability,��multinomial����Ҫ����probability�����ǻ���û���ף����ֻ������������һ������

������ϸ˵��������֮��Ĳ��죬��������ϸ�Ĳ��Խ�����ǳ��ޡ�����⣬multinomial �ǽ�loss�ֳ���������У���softmax���Ǻ���һ���ˡ�����˵��multinomial loss�ǰ����Ͱ�ļ��㷴���ݶȣ���softmax���ǰ���������ֱ�Ӻϲ�Ϊһ����������ˣ��������м�ľ�����ʧ�� ���Ӽ����ȶ��Խ���softmax���ã�multinomial�Ǳ�׼������softmax����һ���Ż��ɡ�
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

  // ���top��һ��������
  top[0]->Reshape(loss_shape);
}

INSTANTIATE_CLASS(LossLayer);

}  // namespace caffe
