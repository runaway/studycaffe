#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/concat_layer.hpp"
#include "caffe/layers/flatten_layer.hpp"
#include "caffe/layers/pooling_layer.hpp"
#include "caffe/layers/split_layer.hpp"
#include "caffe/layers/spp_layer.hpp"

/*
���£�Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition

��Դ��Technicalreport

���⣺ͨ��ͼ���������ʵ��ʶ���еĳ߶��޹��ԣ�

���ߣ�KaimingHe��Xiangyu Zhang, Shaoqing Ren, Jian Sun ,����΢��

��Ҫ���ݣ�
����֮ǰ�Ĵ󲿷�CNNģ�͵�����ͼ���ǹ̶���С�ģ���С������ȣ�������NIPS2012��
��СΪ224X224������ͬ��С������ͼ����Ҫͨ��crop����warp������һ���̶���С��ͼ��
���뵽�����С������Ӿʹ������⣬1.�߶ȵ�ѡ����������ԣ����ڲ�ͬ��Ŀ�꣬�����ʺ�
�ĳߴ��С���ܲ�һ����2.���ڲ�ͬ�ĳߴ��С��ͼ��ͳ���ȵ�ͼ��ǿ�Ʊ任���̶���
��С����ʧ��Ϣ��3.crop��ͼ����ܲ�����������ͼ��warp��ͼ����ܵ��¼����α䡣��
��˵�̶����뵽�����ͼ��Ĵ�С���ܻ�Ӱ�쵽���ǵ�ʶ���ر��Ǽ���׼ȷ�ʣ�

����ƪ�����У���������ÿռ�������ػ���spatial pyramid pooling��SPP����ʵ�ֶ�ͼ
���С�Ͳ�ͬ����ȵĴ��������������µ����磬����SPP-Net�����Բ���ͼ��Ĵ�С��
����ͬ��С���ȵı�ʾ�������������������ڷ���ͼ�����涼ˢ�µļ�¼�������ٶȱȽ�
�죬��30-170������Ϊ֮ǰ�ļ�ⷽ�����ǲ��ã�1.�������ڣ����� 2.�Կ��ܵļ���Ŀ��
��������Ŀ�괰�ڣ������м�ǧ������ÿһ��������ʶ��Ȼ����ѡ�����ֵ��Ϊ��⵽��Ŀ
�ꣻ

�����������磬����ֻ��Ҫ��������ͼ�������ͼ��feature maps��һ�Σ�Ȼ��ػ��Ӵ���
�������������Ͳ����˹̶����ȵı�ʾ������������ѵ���������
 
ΪʲôCNN��Ҫ�̶�����ͼ��Ĵ�С��������ֲ���Ҫ�̶�ͼ��Ĵ�С�����������С�Ǹ�
����ͼ��Ĵ�С��صģ����й̶�����ͼ���С�������ȫ���Ӳ��֣������ǵĶ������ǿ�
��֪����ȫ���Ӳ��ֵĲ����ĸ�������Ҫ�̶��ġ���������֪�����̶���С�������ֻ�Ƿ�
�������������㣨�߲㣩����
 
���������˿ռ�������ػ���spatial pyramidpooling(SPP)������ȥ������̶���С����
�ƣ�Ҳ����˵����SPP��ӵ����һ���������棬SPP��ػ��������Ҳ����̶���С�������
�������Ȼ�����͵���һ��ȫ���Ӳ㡣Ҳ����˵�ھ�����ȫ���Ӳ�֮ǰ�����ǵ�����һ��
�µĲ㣬�����Խ��ܲ�ͬ��С�����뵫�ǲ�����ͬ��С������������Ϳ��Ա������������
��ڴ���Ҫ�����Ǵ�С��ͬ��Ҳ��ʵ����������˵�Ŀ��Խ�����������߶ȣ�
 

����˵������ʽ���������ǵĴ��ԣ����ǵĴ����ܲ�����˵�ȶ����������Ӿ���ͼ�������
����߹�һ��ͬһ�ߴ��ٽ���ʶ�𣬶��ǲ��������������С��ͼ��Ȼ���ٺ��ڽ��д���
 
SSP����˵�ǿռ������ƥ�䣨spatial pyramid matching or SPM����BoW��һ����չ������
һ��ͼƬ����Ϊ�Ӳ�ͬ�ķֱ��ʼ���Ȼ��ۺ���Щ��ͬ�ֱ��ʵ�ͼ�������ѧϰ֮ǰSPMȡ
���˺ܴ�ĳɹ���Ȼ�������ѧϰCNN����֮��ȴ���ٱ��õ���SSP��һЩ�ܺõ�������1.��
���Բ����������ݵĴ�С��������ͬ��С�������������Ͳ��� 2.SPPʹ�ö༶��Ŀռ�飬
Ҳ����˵�����Ա����˺ܴ�һ���ֵķֱ����޹��ԣ�3.SPP���Գػ��Ӳ�ͬ�߶�ͼ����ȡ����
����
 
�Ա���R-CNN��R-CNN����ʱ����Ϊ����ͨ����ͼ��Ĳ�ͬ���򣨼�ǧ����ͨ�������ԣ���ȡ
������ʾ��������ƪ�����У�ֻ��Ҫ���о����һ�Σ�����ͼ�����۴�С����Ȼ������
SPP������ȡ����������ȡ��������������ͬ�ģ�����˵�������˾���Ĵ��������Ա�R-CNN
���˼�ʮ����һ�ٶ౶���ٶȣ�

�ػ��㣨Poolinglayer���ڻ������ڵĽǶ��£�Ҳ���Կ���Ϊ����㣬�����������֮Ϊ
featuremap������ʾ����Ӧ��ǿ�Ⱥ�λ����Ϣ��

������SPP���滻���һ����������ĳػ����У�
 

��ÿһ���ռ�飨bin���У��ػ�ÿһ���˲�������Ӧ������SPP������Ϊ256Mά�ȣ�����
256���˲����ĸ�����M��bin�ĸ���������������Ȼ��M�Ǹ��ݲ�ͬ��ͼ���С��������ģ���
������ͬ����ͼ���С������Ϳ�����ͬ�ˡ�
���ڸ���������ͼ���С�����ǿ����ȼ����������Ҫ�Ŀռ�bin��Ķ��٣��������£�
����һ��224*224��ͼ�������뵽conv5�����Ϊa*a��13*13��������Ҫn*n����Ľ�����
ʱ��ÿ����������Ϊwin=��a/n������Ϊ��a/n��������Ҫl����������ʱ�򣬼����l����
���Ĳ������ںͲ�����Ȼ����Щl�������bin����������Ϊ��һ��ȫ���Ӳ�������
*/

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
LayerParameter SPPLayer<Dtype>::GetPoolingParam(const int pyramid_level,
      const int bottom_h, const int bottom_w, const SPPParameter spp_param) {
  LayerParameter pooling_param;
  int num_bins = pow(2, pyramid_level);

  // ��Ԫ����������������image����ֻ�����Ӿֲ����������������ֲ�����Ұ�����Ĵ�С�������Ϊ kernel size�Ĵ�С��  
  // find padding and kernel size so that the pooling is
  // performed across the entire image
  int kernel_h = ceil(bottom_h / static_cast<double>(num_bins));
  // remainder_h is the min number of pixels that need to be padded before
  // entire image height is pooled over with the chosen kernel dimension
  int remainder_h = kernel_h * num_bins - bottom_h;
  // pooling layer pads (2 * pad_h) pixels on the top and bottom of the
  // image.
  int pad_h = (remainder_h + 1) / 2;

  // similar logic for width
  int kernel_w = ceil(bottom_w / static_cast<double>(num_bins));
  int remainder_w = kernel_w * num_bins - bottom_w;
  int pad_w = (remainder_w + 1) / 2;

  pooling_param.mutable_pooling_param()->set_pad_h(pad_h);
  pooling_param.mutable_pooling_param()->set_pad_w(pad_w);
  pooling_param.mutable_pooling_param()->set_kernel_h(kernel_h);
  pooling_param.mutable_pooling_param()->set_kernel_w(kernel_w);
  pooling_param.mutable_pooling_param()->set_stride_h(kernel_h);
  pooling_param.mutable_pooling_param()->set_stride_w(kernel_w);

  switch (spp_param.pool()) {
  case SPPParameter_PoolMethod_MAX:
    pooling_param.mutable_pooling_param()->set_pool(
        PoolingParameter_PoolMethod_MAX);
    break;
  case SPPParameter_PoolMethod_AVE:
    pooling_param.mutable_pooling_param()->set_pool(
        PoolingParameter_PoolMethod_AVE);
    break;
  case SPPParameter_PoolMethod_STOCHASTIC:
    pooling_param.mutable_pooling_param()->set_pool(
        PoolingParameter_PoolMethod_STOCHASTIC);
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }

  return pooling_param;
}

template <typename Dtype>
void SPPLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  SPPParameter spp_param = this->layer_param_.spp_param();

  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  bottom_h_ = bottom[0]->height();
  bottom_w_ = bottom[0]->width();
  reshaped_first_time_ = false;
  CHECK_GT(bottom_h_, 0) << "Input dimensions cannot be zero.";
  CHECK_GT(bottom_w_, 0) << "Input dimensions cannot be zero.";

  pyramid_height_ = spp_param.pyramid_height();
  split_top_vec_.clear();
  pooling_bottom_vecs_.clear();
  pooling_layers_.clear();
  pooling_top_vecs_.clear();
  pooling_outputs_.clear();
  flatten_layers_.clear();
  flatten_top_vecs_.clear();
  flatten_outputs_.clear();
  concat_bottom_vec_.clear();

  if (pyramid_height_ == 1) {
    // pooling layer setup
    LayerParameter pooling_param = GetPoolingParam(0, bottom_h_, bottom_w_,
        spp_param);
    pooling_layers_.push_back(shared_ptr<PoolingLayer<Dtype> > (
        new PoolingLayer<Dtype>(pooling_param)));
    pooling_layers_[0]->SetUp(bottom, top);
    return;
  }
  // split layer output holders setup
  for (int i = 0; i < pyramid_height_; i++) {
    split_top_vec_.push_back(new Blob<Dtype>());
  }

  // split layer setup
  LayerParameter split_param;
  split_layer_.reset(new SplitLayer<Dtype>(split_param));
  split_layer_->SetUp(bottom, split_top_vec_);

  for (int i = 0; i < pyramid_height_; i++) {
    // pooling layer input holders setup
    pooling_bottom_vecs_.push_back(new vector<Blob<Dtype>*>);
    pooling_bottom_vecs_[i]->push_back(split_top_vec_[i]);

    // pooling layer output holders setup
    pooling_outputs_.push_back(new Blob<Dtype>());
    pooling_top_vecs_.push_back(new vector<Blob<Dtype>*>);
    pooling_top_vecs_[i]->push_back(pooling_outputs_[i]);

    // pooling layer setup
    LayerParameter pooling_param = GetPoolingParam(
        i, bottom_h_, bottom_w_, spp_param);

    pooling_layers_.push_back(shared_ptr<PoolingLayer<Dtype> > (
        new PoolingLayer<Dtype>(pooling_param)));
    pooling_layers_[i]->SetUp(*pooling_bottom_vecs_[i], *pooling_top_vecs_[i]);

    // flatten layer output holders setup
    flatten_outputs_.push_back(new Blob<Dtype>());
    flatten_top_vecs_.push_back(new vector<Blob<Dtype>*>);
    flatten_top_vecs_[i]->push_back(flatten_outputs_[i]);

    // flatten layer setup
    LayerParameter flatten_param;
    flatten_layers_.push_back(new FlattenLayer<Dtype>(flatten_param));
    flatten_layers_[i]->SetUp(*pooling_top_vecs_[i], *flatten_top_vecs_[i]);

    // concat layer input holders setup
    concat_bottom_vec_.push_back(flatten_outputs_[i]);
  }

  // concat layer setup
  LayerParameter concat_param;
  concat_layer_.reset(new ConcatLayer<Dtype>(concat_param));
  concat_layer_->SetUp(concat_bottom_vec_, top);
}

template <typename Dtype>
void SPPLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  // Do nothing if bottom shape is unchanged since last Reshape
  if (num_ == bottom[0]->num() && channels_ == bottom[0]->channels() &&
      bottom_h_ == bottom[0]->height() && bottom_w_ == bottom[0]->width() &&
      reshaped_first_time_) {
    return;
  }
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  bottom_h_ = bottom[0]->height();
  bottom_w_ = bottom[0]->width();
  reshaped_first_time_ = true;
  SPPParameter spp_param = this->layer_param_.spp_param();
  if (pyramid_height_ == 1) {
    LayerParameter pooling_param = GetPoolingParam(0, bottom_h_, bottom_w_,
        spp_param);
    pooling_layers_[0].reset(new PoolingLayer<Dtype>(pooling_param));
    pooling_layers_[0]->SetUp(bottom, top);
    pooling_layers_[0]->Reshape(bottom, top);
    return;
  }
  split_layer_->Reshape(bottom, split_top_vec_);
  for (int i = 0; i < pyramid_height_; i++) {
    LayerParameter pooling_param = GetPoolingParam(
        i, bottom_h_, bottom_w_, spp_param);

    pooling_layers_[i].reset(
        new PoolingLayer<Dtype>(pooling_param));
    pooling_layers_[i]->SetUp(
        *pooling_bottom_vecs_[i], *pooling_top_vecs_[i]);
    pooling_layers_[i]->Reshape(
        *pooling_bottom_vecs_[i], *pooling_top_vecs_[i]);
    flatten_layers_[i]->Reshape(
        *pooling_top_vecs_[i], *flatten_top_vecs_[i]);
  }
  concat_layer_->Reshape(concat_bottom_vec_, top);
}

template <typename Dtype>
void SPPLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (pyramid_height_ == 1) {
    pooling_layers_[0]->Forward(bottom, top);
    return;
  }
  split_layer_->Forward(bottom, split_top_vec_);
  for (int i = 0; i < pyramid_height_; i++) {
    pooling_layers_[i]->Forward(
        *pooling_bottom_vecs_[i], *pooling_top_vecs_[i]);
    flatten_layers_[i]->Forward(
        *pooling_top_vecs_[i], *flatten_top_vecs_[i]);
  }
  concat_layer_->Forward(concat_bottom_vec_, top);
}

template <typename Dtype>
void SPPLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  if (pyramid_height_ == 1) {
    pooling_layers_[0]->Backward(top, propagate_down, bottom);
    return;
  }
  vector<bool> concat_propagate_down(pyramid_height_, true);
  concat_layer_->Backward(top, concat_propagate_down, concat_bottom_vec_);
  for (int i = 0; i < pyramid_height_; i++) {
    flatten_layers_[i]->Backward(
        *flatten_top_vecs_[i], propagate_down, *pooling_top_vecs_[i]);
    pooling_layers_[i]->Backward(
        *pooling_top_vecs_[i], propagate_down, *pooling_bottom_vecs_[i]);
  }
  split_layer_->Backward(split_top_vec_, propagate_down, bottom);
}

INSTANTIATE_CLASS(SPPLayer);
REGISTER_LAYER_CLASS(SPP);

}  // namespace caffe
