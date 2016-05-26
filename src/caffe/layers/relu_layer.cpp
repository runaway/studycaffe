#include <algorithm>
#include <vector>

#include "caffe/layers/relu_layer.hpp"
/*
Relu ������һ���Ƚ����еļ�������Ѿ���ȡ��sigmoid�������������������� z �������Ӷ����ڱ��͡�
���������������sigmoid��tanh�����Ҳ����pre-training�Ļ�����Ϊ gradient vanishing problem �����޷�������ʹ��ReLU����û��������⡣

Ԥѵ�����ô������򻯣���ֹ����ϣ�ѹ�����ݣ�ȥ�����ࣻǿ����������С���ӿ������ٶȡ�

��׼��sigmoid������߱�ϡ���ԣ���Ҫ��һЩ�ͷ�������ѵ����һ��ѽӽ�0���������������Ӷ�����ϡ�����ݣ�����L1��L2���ͷ����ӡ������Ҫ�����޼ල��Ԥѵ����

��ReLU��������������ʽΪ��g(x) = max(0, x)����purelin�����߰档��������������������ֵС��0������������0�����򱣳�ԭ����ֵ���䡣

����һ�ּ򵥴ֱ���ǿ��ĳЩ����Ϊ0�ķ�����Ȼ����ʵ��֤����ѵ�����������ȫ�߱��ʶȵ�ϡ���ԡ�����ѵ����Ŀ��ӻ�Ч���ʹ�ͳ��ʽԤѵ������Ч�������ƣ���Ҳ˵����ReLU�߱������ʶ�ϡ���������

�������ֳ�����relu���޸İ�prelu�����ﲻ�����ܡ�
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
