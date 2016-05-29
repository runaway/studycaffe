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
    // ���������б��Ϊ1��С��0б��Ϊnegative_slope��
    top_data[i] = std::max(bottom_data[i], Dtype(0))
        + negative_slope * std::min(bottom_data[i], Dtype(0));
  }
}
/*
����Backward_cpu()������˵���� 
propagate_down����caffe.proto����һ��˵����Specifies on which bottoms the backpropagation should be skipped.The size must be either 0 or equal to the number of bottoms. propagate_down����������bottom���ݶ���صġ������ͨ��������ϣ����ݶȶ�������ڲ���weights���Եģ����ǣ���caffe��Ϊʲô������bottom�ĵ�����һ˵�أ�����ԭ������caffe��ʵ�֡���ʽ��1���У���(l)��ʵ������ʧ�������ڵ�ǰ������루bottom����ƫ����������� propagate_down���Ǽ��������(l)�Ŀ����������ɹ�ʽ�Ϳ���֪���������(l)��caffe��BPʵ�����Ƿǳ���Ҫ��

caffe��BP��ʵ�֣� caffe��ģ�黯�ǳ�ǿ������W��X��������ͣ����������Լ�pooling���ֿ��ˣ�Ҳ����˵caffe���conv_layerֻ��һ������������㣬��û�м������㣬���ҷֱ��Ӧ��caffe���conv_layer, Relu_layer, �Լ�pooling_layer���빫ʽ��3����Ӧ��������Relu_layerΪ����˵��һ��caffe��BPʵ�֣���ʽ�еĦ�(l+1)��ʵ��top_diff��Ӧ����(l)��ʵ��bottom_diff��Ӧ����top_diff = top[0]->cpu_diff()��bottom_diff = bottom[0]->mutable_cpu_diff()������ΪRelu_layer��û��weight������ˣ������Թ�ʽ��3������Ϊ��(l)=��(l+1)?f��(z(l))�� ��(l+1)������һ��,f��(z(l))������е�forѭ�������Ӧ��������conv_layer����ʽ��3������Ϊ��(l)=(W(l))T��(l+1)�����Բ鿴ConvolutionLayer::Backward_cpu()����֤����Ӧ�ڹ�ʽ��4��������ǹ���weights����˵��ݶȣ�a(l)�뵱ǰ������bottom��ء���֮������caffe�ĸ߶�ģ���ԣ���BPʵ�ֵ��ݶ��й���bottom�ģ�Ҳ�й���weights�ġ�

��ʽ��2����ʵ�ǹ�ʽ��1����JȡEuclideanLoss��������(y?a(nl))����top_diff�����ڴˣ��������happynear������˵�ģ�CNN���� = FP������ͼ�񣩣� diff = ԭʼ���� - CNN������ loss = lossfunction��diff�����µ�grad=BP��loss��
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
