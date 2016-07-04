// TODO (sergeyk): effect should not be dependent on phase. wasted memcpy.

#include <vector>

#include "caffe/layers/dropout_layer.hpp"
#include "caffe/util/math_functions.hpp"

/*
dropout layer��Ŀ����Ϊ�˷�ֹCNN ����ϡ���ôΪʲô������Ч�ķ�ֹ������أ�
���ȣ�������������ֻѵ��һ���ض������磬���������������ʱ�򣬿��ܳ��������ѵ������ϵĺܺã���ѵ������loss��С�������Ƕ���֤������ϳ̶Ⱥܲ����������ԣ����������������뷨���ɲ�������ÿ�ε��������ȥ�������������weights������������������ԾͿ�����������generalize �����������Ծ�����dropout ��
��ѵ����ʱ������ֻ��Ҫ��һ���ĸ��ʣ�retaining probability��p ����weight layer �Ĳ�����������������������������Ϊ�˴θ��µ�Ŀ�����硣���������������������n����������ô���ǿ��õ����������Ϊ 2^n �� ���ң���n�ܴ�ʱ��ÿ�ε������� ʹ�õ�����������ϲ����ظ����Ӷ�������ĳһ�����类���ֵ���ϵ�ѵ�����ϡ�
��ô���Ե�ʱ����ô���أ� һ����naive�ķ����ǣ����ǰ� 2^n �������綼���������ԣ�Ȼ����ĳ�� voting ���ƽ����н�����һ�£�����˵ƽ��һ���£���Ȼ��õ����յĽ�������ǣ�����nʵ����̫���ˣ����ַ���ʵ������ȫ�����У� 
�������������������һ�����µĹ��Ʋ��͵��ˣ��Ҵ�2^n�����������ѡȡ m �����������ԣ��������ĳ��voting ���Ƶõ����յ�Ԥ�����������뷨��Ȼ���У���m�ܴ�ʱ����ԶС��2^nʱ���ܹ��ܺõıƽ�ԭ2^n��������������Ԥ���������ǣ���û�и��õİ취�أ� of course���Ǿ���dropout �Դ��Ĺ��ܣ��ܹ�ͨ��һ�β��Եõ��ƽ���ԭ2^n���������������Ԥ�������� 
��Ȼѵ����ʱ������ʹ����dropout�� �����ڲ���ʱ�����ǲ�ʹ��dropout ����������Ĳ������κζ�������ʱdropout layer�൱�ڽ���ʲô�����ʲô����Ȼ�󣬰Ѳ���ʱdropout layer���������ѵ��ʱʹ�õ�retaining probability  p ����ʱdropout layer�൱�ڰѽ����Ķ�������p������ϸ����������������������أ����� ��ʵ�ϣ����������ڲ���ʱ�����κεĲ�����������������˵��dropout layer �ѽ����Ķ���ԭ�������������ͳ�������£�����ʱ ÿ�� dropout layer�������ѵ��ʱ���������ˡ���1 - p��*100��%  units ������� �� ��p*100��% ��units �ĺ�  ��ͬѵ��ʱ��������õ�������������һ�£�����1 - p��*100��%  ��units�ĺ�  �Ǳ���Ӧ���ӵ��������ڲ��Խ׶α����������ġ����ԣ�Ϊ��ʹ��dropout layer ��һ��������ѵ��ʱ������ͬ�ġ����塱�͡���������������Ҫ�Բ���ʱ��αdropout layer����������²�����룩�� rescale�� ����һ��p����ʾ����sum��ֻ����ô��ĸ��ʣ�������ô��Ĳ��ֱ�����������������ֻҪһ�β��ԣ���ԭ2^n��������Ĳ���ȫ�����ǽ����ˣ��������� rescale ��֤�˺���һ���������Ȼ������Ӧ�������������������

����x��dropout layer�����룬y��dropout layer�������W����һ�������weight parameters�� ����retaining probability Ϊp �����õ���weight parameter�Ӽ���������Ķ����ù�ʽ��ʾ������bias����
train��  
test: 
����һ��д�����ʱ��������ֱ����testʱ��   �� ���ֱ��ʽ����where  �� 
������Ǿ���ѵ����ʱ���ֱ��ѵ��  �� ����ѵ��ʱ����һ����ʽ����Ϊ    �� ����dropout���������p �ٽ���ѵ���������õ���ѵ���õ���weight ��������  �����Ե�ʱ����˲�ʹ��dropout�⣬����Ҫ�����κ�rescale��Caffe ��Lasagne ����Ĵ����������д�ġ�

�ο����ף� Improving Neural Networks with Dropout�� Hintonѧ����һƬthesis
       Dropout: A Simple Way to Prevent Neural Networks from Overfitting
*/

namespace caffe {

template <typename Dtype>
void DropoutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  threshold_ = this->layer_param_.dropout_param().dropout_ratio();
  DCHECK(threshold_ > 0.);
  DCHECK(threshold_ < 1.);
  scale_ = 1. / (1. - threshold_);
  uint_thres_ = static_cast<unsigned int>(UINT_MAX * threshold_);
}

template <typename Dtype>
void DropoutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  // Set up the cache for random number generation
  // ReshapeLike does not work because rand_vec_ is of Dtype uint
  rand_vec_.Reshape(bottom[0]->shape());
}

template <typename Dtype>
void DropoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) 
{
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    unsigned int* mask = rand_vec_.mutable_cpu_data();
    const int count = bottom[0]->count();
    
    if (this->phase_ == TRAIN) 
    {
        // Create random numbers
        caffe_rng_bernoulli(count, 1. - threshold_, mask);

        for (int i = 0; i < count; ++i) 
        {
            top_data[i] = bottom_data[i] * mask[i] * scale_;
        }
    } 
    else 
    {
        caffe_copy(bottom[0]->count(), bottom_data, top_data);
    }
}

template <typename Dtype>
void DropoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) 
{
    if (propagate_down[0]) 
    {
        const Dtype* top_diff = top[0]->cpu_diff();
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

        if (this->phase_ == TRAIN) 
        {
            const unsigned int* mask = rand_vec_.cpu_data();
            const int count = bottom[0]->count();

            for (int i = 0; i < count; ++i) 
            {
                bottom_diff[i] = top_diff[i] * mask[i] * scale_;
            }
        } 
        else 
        {
            caffe_copy(top[0]->count(), top_diff, bottom_diff);
        }
    }
}


#ifdef CPU_ONLY
STUB_GPU(DropoutLayer);
#endif

INSTANTIATE_CLASS(DropoutLayer);
REGISTER_LAYER_CLASS(Dropout);

}  // namespace caffe
