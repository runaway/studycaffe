#include <vector>

#include "caffe/layers/conv_layer.hpp"


/*
Vision Layers:
 

ͷ�ļ�./include/caffe/vision_layers.hpp

vision layersͨ��ȡͼ��Ϊ���룬��������ͼ����Ϊ�����ʵ���е��͵�ͼ�����ֻ��һ����ɫͨ��(c = 1)��������һ���Ҷ�ͼ���У���������ͨ��(c = 3)����һ��RGBͼ���С��������һ��ͼ����������������Ŀռ�ṹ��ͨ��һ��ͼ���и߶�h > 1�����w > 1�����2D����ͼ����Ȼ��������δ������롣�ر�أ������vision layersͨ���������һЩ����Ӧ��һ������Ĳ�����������Ӧ��������Ա�����������layers���������⣩��������Ŀռ�ṹ����������Ϊһ���������������ά��Ϊchw��

 

Convolution layer��

layer���ͣ�Convolution
CPUʵ�֣�./src/caffe/layers/conv_layer.cpp
CUDA GPUʵ�֣�./src/caffe/layers/conv_layer.cu
����(ConvolutionParameter convolution_param)
����Ҫ���
num_output(c_o): �˲�������
kernel_size (or kernel_h and kernel_w): ÿ���˲����ĸߺͿ�
ǿ���Ƽ���
weight_filter [default type: 'constant' value: 0]
��ѡ��
bias_term [default true]: �Ƿ�ѧϰ��Ӧ��һ��biase���˲������
pad (or pad_h and pad_w) [default 0]: ָ��������ͼ���ÿ����������ӵ�������Ŀ
stride (or stride_h and stride_w) [default 1]: ָ��Ӧ���˲�����ͼ��ʱ�˲����ļ��
group (g) [default 1]: ��� g > 1����������ÿ���˲������ӵ�������Ӽ����ر�أ���������ͨ������Ϊg�飬��i������������ӵ���i�����롣
���룺 n * c_i * h_i * w_i
�����n * c_o * h_o * w_o������h_o = (h_i + 2 * pad_h - kernel_h) / stride_h + 1��w_o�ɵ����ƽ����
���ӣ�
 

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
20
21
22
23
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  # learning rate and decay multipliers for the filters
  param { lr_mult: 1 decay_mult: 1 }
  # learning rate and decay multipliers for the biases
  param { lr_mult: 2 decay_mult: 0 }
  convolution_param {
    num_output: 96     # learn 96 filters
    kernel_size: 11    # each filter is 11x11
    stride: 4          # step 4 pixels between each filter application
    weight_filler {
      type: "gaussian" # initialize the filters from a Gaussian
      std: 0.01        # distribution with stdev 0.01 (default mean: 0)
    }
    bias_filler {
      type: "constant" # initialize the biases to zero (0)
      value: 0
    }
  }
}
Convolution layer�������ͼ���һ���ѧϰ���˲�����ÿ���˲�����Ӧ�ز������ͼ���һ��feature map��

*/
namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

// ��ÿһ��ͼ���о�����㣬Ȼ��洢�� �ȼ���weight, �ټ���bias
template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) 
{
    const Dtype* weight = this->blobs_[0]->cpu_data();

    // blobs_ �����洢��ѧϰ�Ĳ���blobs_[0] ��weight��blobs_[1]��bias
    for (int i = 0; i < bottom.size(); ++i) 
    {
        // �����iΪ����bottom�ĸ�����������ٸ�bottom�Ͳ�����Ӧ��������� top��
        const Dtype* bottom_data = bottom[i]->cpu_data();
        Dtype* top_data = top[i]->mutable_cpu_data();

        // num_ = batchsize  
        for (int n = 0; n < this->num_; ++n) 
        {
            // �����forward_cpu_gemm����  
            // �������top_data[n * this->top_dim_] =  
            // weights X bottom_data[n * this->bottom_dim_]  
            // �������һ��ͼ������ݣ���Ӧ�������ͼ����֮���λ�� 
            this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
            top_data + n * this->top_dim_); // ����������֮������
            
            if (this->bias_term_) 
            {
                const Dtype* bias = this->blobs_[1]->cpu_data();
                this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
            }
        }
    }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  
  for (int i = 0; i < top.size(); ++i) {

    // ��һ�㴫�����ĵ���
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();

    // ������һ��ĵ���
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        } //��weight ���㵼������������weight��
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        } //��bottom���ݼ��㵼����������һ�㣩
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
