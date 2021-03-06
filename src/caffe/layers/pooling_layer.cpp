#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

/*
Pooling layers：

layer类型：Pooling
CPU实现：./src/caffe/layers/pooling_layer.cpp
CUDA GPU实现：./src/caffe/layers/pooling_layer.cu
参数(PoolingParameter pooling_param)
必须要求的
kernel_size (or kernel_h and kernel_w): 每个滤波器的高和宽
强烈推荐的
weight_filter [default type: 'constant' value: 0]
可选的
pool [default MAX]: pooling的方法，包括MAX, AVE, or STOCHASTIC
pad (or pad_h and pad_w) [default 0]: 指定在输入图像的每个边隐含添加的像素数目
stride (or stride_h and stride_w) [default 1]: 指定应用滤波器到图像时滤波器的间隔
输入： n * c * h_i * w_i
输出：n * c * h_o * w_o，其中h_o = (h_i + 2 * pad_h - kernel_h) / stride_h + 1，w_o可得类似结果。
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
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 3 # pool over a 3x3 region
    stride: 2      # step two pixels (in the bottom blob) between pooling regions
  }
}

*/

namespace caffe 
{

using std::min;
using std::max;

template <typename Dtype>
void PoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
    PoolingParameter pool_param = this->layer_param_.pooling_param();

    if (pool_param.global_pooling()) 
    {
        CHECK(!(pool_param.has_kernel_size() ||
        pool_param.has_kernel_h() || pool_param.has_kernel_w()))
        << "With Global_pooling: true Filter size cannot specified";
    } 
    else 
    {
        CHECK(!pool_param.has_kernel_size() !=
        !(pool_param.has_kernel_h() && pool_param.has_kernel_w()))
        << "Filter size is kernel_size OR kernel_h and kernel_w; not both";

        CHECK(pool_param.has_kernel_size() ||
        (pool_param.has_kernel_h() && pool_param.has_kernel_w()))
        << "For non-square filters both kernel_h and kernel_w are required.";
    }
    
    CHECK((!pool_param.has_pad() && pool_param.has_pad_h()
    && pool_param.has_pad_w())
    || (!pool_param.has_pad_h() && !pool_param.has_pad_w()))
    << "pad is pad OR pad_h and pad_w are required.";
    
    CHECK((!pool_param.has_stride() && pool_param.has_stride_h()
    && pool_param.has_stride_w())
    || (!pool_param.has_stride_h() && !pool_param.has_stride_w()))
    << "Stride is stride OR stride_h and stride_w are required.";
    
    global_pooling_ = pool_param.global_pooling();

    // 全局pooling
    if (global_pooling_) 
    {
        kernel_h_ = bottom[0]->height();
        kernel_w_ = bottom[0]->width();
    } 
    else 
    {
        if (pool_param.has_kernel_size()) 
        {
            kernel_h_ = kernel_w_ = pool_param.kernel_size();
        } 
        else 
        {
            // 用户自定义的kernel大小
            kernel_h_ = pool_param.kernel_h();
            kernel_w_ = pool_param.kernel_w();
        }
    }
    
    CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
    CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
    
    if (!pool_param.has_pad_h()) 
    {
        pad_h_ = pad_w_ = pool_param.pad();
    } 
    else 
    {
        // 填充
        pad_h_ = pool_param.pad_h();
        pad_w_ = pool_param.pad_w();
    }

    // 步长
    if (!pool_param.has_stride_h()) 
    {
        stride_h_ = stride_w_ = pool_param.stride();
    } 
    else 
    {
        stride_h_ = pool_param.stride_h();
        stride_w_ = pool_param.stride_w();
    }
    
    if (global_pooling_) 
    {
        CHECK(pad_h_ == 0 && pad_w_ == 0 && stride_h_ == 1 && stride_w_ == 1)
        << "With Global_pooling: true; only pad = 0 and stride = 1";
    }

    // 初始化一些参数
    if (pad_h_ != 0 || pad_w_ != 0) 
    {
        CHECK(this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_AVE
        || this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_MAX)
        << "Padding implemented only for average and max pooling.";

        CHECK_LT(pad_h_, kernel_h_);
        CHECK_LT(pad_w_, kernel_w_);
    }
}

template <typename Dtype>
void PoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
    CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
    
    channels_ = bottom[0]->channels();
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    
    if (global_pooling_) 
    {
        kernel_h_ = bottom[0]->height();
        kernel_w_ = bottom[0]->width();
    }

    // pooling之后的height和width
    pooled_height_ = static_cast<int>(ceil(static_cast<float>(
      height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
    pooled_width_ = static_cast<int>(ceil(static_cast<float>(
      width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
    
    if (pad_h_ || pad_w_) 
    {
        // If we have padding, ensure that the last pooling starts strictly
        // inside the image (instead of at the padding); otherwise clip the last.
        if ((pooled_height_ - 1) * stride_h_ >= height_ + pad_h_) 
        {
            --pooled_height_;
        }
        
        if ((pooled_width_ - 1) * stride_w_ >= width_ + pad_w_) 
        {
            --pooled_width_;
        }
        
        CHECK_LT((pooled_height_ - 1) * stride_h_, height_ + pad_h_);
        CHECK_LT((pooled_width_ - 1) * stride_w_, width_ + pad_w_);
    }

    // 输出top blob 的shape
    top[0]->Reshape(bottom[0]->num(), channels_, pooled_height_,
      pooled_width_);
    
    if (top.size() > 1) 
    {
        top[1]->ReshapeLike(*top[0]);
    }

    // max pooling 反向求导时要用到取最大值的位置，max_idx_就是记录pooling过程
    // 中取max value 的index ，它是一个int型的blob 和输出top具有相同的shape
    // If max pooling, we will initialize the vector index part.
    if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_MAX && top.size() == 1) 
    {
        max_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
            pooled_width_);
    }

    // 类似于max pooling
    // If stochastic pooling, we will initialize the random index part.
    if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_STOCHASTIC) 
    {
        rand_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
          pooled_width_);
    }
}

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <typename Dtype>
void PoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    const int top_count = top[0]->count();
    // We'll output the mask to top[1] if it's of size >1.
    const bool use_top_mask = top.size() > 1;
    int* mask = NULL;  // suppress warnings about uninitalized variables
    Dtype* top_mask = NULL;
    
    // Different pooling methods. We explicitly do the switch outside the for
    // loop to save time, although this results in more code.
    switch (this->layer_param_.pooling_param().pool()) 
    {
    case PoolingParameter_PoolMethod_MAX: // 最大采样方法

        // Initialize
        if (use_top_mask) 
        {
            top_mask = top[1]->mutable_cpu_data();
            caffe_set(top_count, Dtype(-1), top_mask);
        } 
        else 
        {
            // 模板类Blob的mutable_cpu_diff()方法中使用了强制类型转换static_cast<Dtype*>()  
            //（*1）设为负无穷
            mask = max_idx_.mutable_cpu_data();
            caffe_set(top_count, -1, mask);
        }
        
        // FLT_MAX在头文件#include <cfloat>中定义 
        caffe_set(top_count, Dtype(-FLT_MAX), top_data);

        // The main loop
        for (int n = 0; n < bottom[0]->num(); ++n) 
        {
            for (int c = 0; c < channels_; ++c) 
            {
                for (int ph = 0; ph < pooled_height_; ++ph) 
                {

                    for (int pw = 0; pw < pooled_width_; ++pw) 
                    {
                        // 这四个量给出未pooling矩阵中确定pooling区域的两个顶点。
                        int hstart = ph * stride_h_ - pad_h_;
                        int wstart = pw * stride_w_ - pad_w_;
                        int hend = min(hstart + kernel_h_, height_);
                        int wend = min(wstart + kernel_w_, width_);
                        hstart = max(hstart, 0);
                        wstart = max(wstart, 0); // 一般情况下从0开始，而不是从负下标开始 

                        // 池化后的（输出）特征图中元素的位置索引   
                        const int pool_index = ph * pooled_width_ + pw; 

                        // caffe 数据存储是一维数组的形式
                        // ph为pooling后输出top的height index，pool_index为对应一维数组index。
                        for (int h = hstart; h < hend; ++h) 
                        {

                            // 找出核范围内最大
                            for (int w = wstart; w < wend; ++w) 
                            {
                                // 输入特征图中元素的位置索引 
                                const int index = h * width_ + w;

                                // 对应一维数组的index
                                if (bottom_data[index] > top_data[pool_index]) 
                                {
                                    top_data[pool_index] = bottom_data[index];

                                    // 由（*1）可知该循环将bottom中pooling区域（kernel的大小）的最大值放到对应top
                                    if (use_top_mask) 
                                    {
                                        top_mask[pool_index] = static_cast<Dtype>(index);
                                    } 
                                    else 
                                    {
                                        // 每次Max_pooling操作最大元素的位置索引  
                                        // 记录top得到的max value在bottom中的index
                                        mask[pool_index] = index;
                                    }
                                }
                            }
                        }
                    }
                }

                // 每次通过offset来确定新的bottom_data地址，offset()函数返回的其
                // 实仅仅是一个整数，大小为一个channel的元素的个数。也就是这样一
                // 个channel一个channel得遍历整个Blob。  
                // 指针移动到下一个channel。注意代码这里的位置。采样是针对每个channel的。
                // compute offset
                bottom_data += bottom[0]->offset(0, 1);
                top_data += top[0]->offset(0, 1);

                if (use_top_mask) 
                {
                    top_mask += top[0]->offset(0, 1);
                } 
                else 
                {
                    // 取下一个channel的mask
                    mask += top[0]->offset(0, 1);
                }
            }
        }
    break;
    
    case PoolingParameter_PoolMethod_AVE:

        for (int i = 0; i < top_count; ++i) 
        {
            // 将top初始化为0
            top_data[i] = 0;
        }
        
        // The main loop
        for (int n = 0; n < bottom[0]->num(); ++n) 
        {
            for (int c = 0; c < channels_; ++c) 
            {
                for (int ph = 0; ph < pooled_height_; ++ph) 
                {
                    for (int pw = 0; pw < pooled_width_; ++pw) 
                    {
                        // pooling 区域的element个数
                        int hstart = ph * stride_h_ - pad_h_;
                        int wstart = pw * stride_w_ - pad_w_;
                        int hend = min(hstart + kernel_h_, height_ + pad_h_);
                        int wend = min(wstart + kernel_w_, width_ + pad_w_);
                        int pool_size = (hend - hstart) * (wend - wstart);
                        hstart = max(hstart, 0);
                        wstart = max(wstart, 0);
                        hend = min(hend, height_);
                        wend = min(wend, width_);
                        
                        for (int h = hstart; h < hend; ++h) 
                        {
                            // 核范围内算平均
                            for (int w = wstart; w < wend; ++w) 
                            {
                                // 将pooling区域的element个数加起来
                                top_data[ph * pooled_width_ + pw] +=
                                    bottom_data[h * width_ + w];
                            }
                        }
                        
                        // 求平均值
                        top_data[ph * pooled_width_ + pw] /= pool_size;
                    }
                }
                
                // 移动到下一个channel
                // compute offset
                bottom_data += bottom[0]->offset(0, 1);
                top_data += top[0]->offset(0, 1);
            }
        }
        
        break;
        
        case PoolingParameter_PoolMethod_STOCHASTIC:
            
            NOT_IMPLEMENTED;
            
        break;
        
        default:
            LOG(FATAL) << "Unknown pooling method.";
    }
}

template <typename Dtype>
void PoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{

    // propagate_down的妙处于此！caffe.proto里面也有一个相同名字的定义      
    if (!propagate_down[0]) 
    {
        return;
    }

    const Dtype* top_diff = top[0]->cpu_diff();

    // 模板类Blob的mutable_cpu_diff()方法中使用了强制类型转换static_cast<Dtype*>()    
    // 初始化bottom_diff 为0
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

    // Different pooling methods. We explicitly do the switch outside the for
    // loop to save time, although this results in more codes.
    caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);

    // We'll output the mask to top[1] if it's of size >1.
    const bool use_top_mask = top.size() > 1;
    const int* mask = NULL;  // suppress warnings about uninitialized variables
    const Dtype* top_mask = NULL;

    switch (this->layer_param_.pooling_param().pool()) 
    {

    case PoolingParameter_PoolMethod_MAX:

        // The main loop
        if (use_top_mask) 
        {
            top_mask = top[1]->cpu_data();
        } 
        else 
        {
            // 取数据成员max_idx_的地址  
            mask = max_idx_.cpu_data();
        }
        
        for (int n = 0; n < top[0]->num(); ++n) 
        {
            for (int c = 0; c < channels_; ++c) 
            {
                for (int ph = 0; ph < pooled_height_; ++ph) 
                {
                    for (int pw = 0; pw < pooled_width_; ++pw) 
                    {
                        // 这里的index是前向传播池化后的特征图中元素的位置索引  
                        const int index = ph * pooled_width_ + pw;
                        const int bottom_index =
                            use_top_mask ? top_mask[index] : mask[index];

                        // 计算“敏感值”分布
                        bottom_diff[bottom_index] += top_diff[index];
                    }
                }
                
                // 采样层输出的残传播给输入。由于是最大采样方法，输出存的都是输入范围内最大的值，所以残差传播的时候也只有范围内最大的值受影响
                bottom_diff += bottom[0]->offset(0, 1);

                // 指向下一个channel
                top_diff += top[0]->offset(0, 1);

                if (use_top_mask) 
                {
                    top_mask += top[0]->offset(0, 1);
                } 
                else 
                {
                    mask += top[0]->offset(0, 1);
                }
            }
        }
    break;

    case PoolingParameter_PoolMethod_AVE:

        // The main loop
        for (int n = 0; n < top[0]->num(); ++n) 
        {
            for (int c = 0; c < channels_; ++c) 
            {
                for (int ph = 0; ph < pooled_height_; ++ph) 
                {
                    for (int pw = 0; pw < pooled_width_; ++pw) 
                    {
                        int hstart = ph * stride_h_ - pad_h_;
                        int wstart = pw * stride_w_ - pad_w_;
                        int hend = min(hstart + kernel_h_, height_ + pad_h_);
                        int wend = min(wstart + kernel_w_, width_ + pad_w_);
                        int pool_size = (hend - hstart) * (wend - wstart);
                        hstart = max(hstart, 0);
                        wstart = max(wstart, 0);
                        hend = min(hend, height_);
                        wend = min(wend, width_);

                        // 遍历pooling区域
                        for (int h = hstart; h < hend; ++h) 
                        {
                            for (int w = wstart; w < wend; ++w) 
                            {
                                // 采样层输出的残差传播给输入，由于是平均采样，所以权重都是
                                // 反向传播时各层间“误差敏感”总和不变，所以对应每个值需要平摊
                                bottom_diff[h * width_ + w] +=
                                  top_diff[ph * pooled_width_ + pw] / pool_size;
                            }
                        }
                    }
                }
                
                // offset
                bottom_diff += bottom[0]->offset(0, 1);

                // 指向下一个channel
                top_diff += top[0]->offset(0, 1);
            }
        }
    break;

    case PoolingParameter_PoolMethod_STOCHASTIC:

        NOT_IMPLEMENTED;
    break;

    default:
        LOG(FATAL) << "Unknown pooling method.";
        
    }
}


#ifdef CPU_ONLY
STUB_GPU(PoolingLayer);
#endif

INSTANTIATE_CLASS(PoolingLayer);

}  // namespace caffe
