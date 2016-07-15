#include <vector>

#include "caffe/layers/conv_layer.hpp"


/*
Vision Layers:
 

头文件./include/caffe/vision_layers.hpp

vision layers通常取图像为输入，产生其他图像作为输出。实际中典型的图像可能只有一个颜色通道(c = 1)，例如在一个灰度图像中，或者三个通道(c = 3)，在一个RGB图像中。但是这里，一个图像的显著特征是它的空间结构，通常一个图像有高度h > 1，宽度w > 1。这个2D几何图形自然导致了如何处理输入。特别地，大多数vision layers通过对输入的一些区域应用一个特殊的操作，产生相应的输出。对比来看，其他layers（少数例外）忽略输入的空间结构，将输入视为一个大的向量，向量维度为chw。

 

Convolution layer：

layer类型：Convolution
CPU实现：./src/caffe/layers/conv_layer.cpp
CUDA GPU实现：./src/caffe/layers/conv_layer.cu
参数(ConvolutionParameter convolution_param)
必须要求的
num_output(c_o): 滤波器数量
kernel_size (or kernel_h and kernel_w): 每个滤波器的高和宽
强烈推荐的
weight_filter [default type: 'constant' value: 0]
可选的
bias_term [default true]: 是否学习和应用一组biase到滤波器输出
pad (or pad_h and pad_w) [default 0]: 指定在输入图像的每个边隐含添加的像素数目
stride (or stride_h and stride_w) [default 1]: 指定应用滤波器到图像时滤波器的间隔
group (g) [default 1]: 如果 g > 1，我们限制每个滤波器连接到输入的子集。特别地，输入和输出通道被分为g组，第i组输出仅仅连接到第i组输入。
输入： n * c_i * h_i * w_i
输出：n * c_o * h_o * w_o，其中h_o = (h_i + 2 * pad_h - kernel_h) / stride_h + 1，w_o可得类似结果。
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
Convolution layer卷积输入图像和一组可学习的滤波器，每个滤波器对应地产生输出图像的一个feature map。

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

// 对每一张图进行卷积计算，然后存储。 先计算weight, 再计算bias
template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) 
{
    const Dtype* weight = this->blobs_[0]->cpu_data();

    // blobs_ 用来存储可学习的参数blobs_[0] 是weight，blobs_[1]是bias
    for (int i = 0; i < bottom.size(); ++i) 
    {
        // 这里的i为输入bottom的个数，输入多少个bottom就产生相应个数的输出 top。
        const Dtype* bottom_data = bottom[i]->cpu_data();
        Dtype* top_data = top[i]->mutable_cpu_data();

        // num_ = batchsize  
        for (int n = 0; n < this->num_; ++n) 
        {
            // 基类的forward_cpu_gemm函数  
            // 计算的是top_data[n * this->top_dim_] =  
            // weights X bottom_data[n * this->bottom_dim_]  
            // 输入的是一幅图像的数据，对应的是这幅图像卷积之后的位置 
            this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
            top_data + n * this->top_dim_); // 计算卷积操作之后的输出
            
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

    // 上一层传下来的导数
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();

    // 传给下一层的导数
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
        } //对weight 计算导数（用来更新weight）
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        } //对bottom数据计算导数（传给下一层）
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
