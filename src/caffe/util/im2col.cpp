#include <vector>

#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe 
{
// 这个函数就是一个波尔型函数，条件是0<=a<b为真，函数也可以输入一个参数代替两个  
// Function uses casting from int to unsigned to compare if value of
// parameter a is greater or equal to zero and lower than value of
// parameter b. The b parameter is of type signed and is always positive,
// therefore its value is always lower than 0x800... where casting
// negative value of a parameter converts it to value higher than 0x800...
// The casting allows to use one condition instead of two.
inline bool is_a_ge_zero_and_a_lt_b(int a, int b) 
{
    return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

// 调用im2col_cpu的时候输入的参数为
// im2col_cpu(一幅图像，输入图像的channel, 输入图像的height, 输入图像的width, kernel的height, kernel的width, pad的height, pad的width, stride的height， stride的width)

// 将输入的图像首先进行虚假pad(啥叫虚假填充，就是实际没填充，但是目标图像中有了
// 填充的0)  
// 填充这一步，我们在原图像上并没有做pad,只是在处理后的图像上加上了pad的值  
// 然后按照channel*kernel_h*kernel_w一列，将一个channel x kernel_h x kernel_w 
// 大小的图像块变成一个列。  
// 有多少个这样的列呢，这就可以用公式进行计算  
// 列数 = [(图像高度+2*填充高度-kernel高度)/stride高度+1] * [(图像宽度+2*填充宽
// 度-kernel宽度)/stride宽度+1]  
// 这个行数就是一个kernel大小的图像块的维度  
// 这个列数实际上就是kernel在图像上滑动的次数
template <typename Dtype>
void im2col_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    Dtype* data_col) 
{
    const int output_h = (height + 2 * pad_h -
    (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int output_w = (width + 2 * pad_w -
    (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    const int channel_size = height * width;

    // 遍历一个kernel大小的图像
    for (int channel = channels; channel--; data_im += channel_size) 
    {
        // 下面三行是计算在kernel大小的图像上面的位置  
        //  c_im     h_offset     w_offset  

        // 遍历卷积之后的图像的上面的每一个像素  
        for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) 
        {
            for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) 
            {

                // 计算卷积之后的图像与卷积之前的图像的位置  
   
                // 卷积之后的图像与卷积之前的图像像素所对应的位置  
                // 卷积之后的像素为h和w那么所对应的原图像的位置为 [h * stride_h - pad_h,   h * stride_h - pad_h+kernel_h]以及  
                // [w * stride_w - pad_w,   w * stride_w - pad_w+kernel_w]  
                int input_row = -pad_h + kernel_row * dilation_h;

                for (int output_rows = output_h; output_rows; output_rows--) 
                {
                    // 如果符合input_row>=height运行循环里面的代码,然后去掉大于height的部分  
                    if (!is_a_ge_zero_and_a_lt_b(input_row, height)) 
                    {
                        for (int output_cols = output_w; output_cols; output_cols--) 
                        {
                            *(data_col++) = 0;
                        }
                    } 
                    else 
                    {
                        int input_col = -pad_w + kernel_col * dilation_w;

                        for (int output_col = output_w; output_col; output_col--) 
                        {
                            // 如果符合input_col<width运行循环里面的代码,然后将这个位置的kernel的地址给了data_col,否则去掉大于width的部分  
                            if (is_a_ge_zero_and_a_lt_b(input_col, width)) 
                            {
                                *(data_col++) = data_im[input_row * width + input_col];
                            } 
                            else 
                            {
                                *(data_col++) = 0;
                            }
                            
                            input_col += stride_w;
                        }
                    }
                    
                    input_row += stride_h;
                }
            }
        }
    }
}

// Explicit instantiation
template void im2col_cpu<float>(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    float* data_col);
template void im2col_cpu<double>(const double* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    double* data_col);
/*
上面介绍了二维卷积，那么我们就趁热打铁，再看看n维通用卷积是如何实现的
接下来介绍n维通用的卷积的具体实现
n维卷积的实现与二维卷积的实现很类似，只不过对应的变量进行了变化，你只需要找到对
应就可以很快理解

d_offset 对应于im2col中的h_offset和w_offset是一个输入图像的channel 乘以
kernel_size大小的图像块的偏移量(kernel_size下面的代码有定义)
d_iter对应于im2col中内层for循环的h和w，是经过im2colnd处理过的col_buff中的偏移
d_pad对应于im2col中内层for循环的h_pad和w_pad，是输入的原始图像中的偏移

作者还将im2colnd和col2imnd合并到一起实现了，通过const bool im2col来判断是im2col
还是col2im
*/

// n维通用im2col以及col2im的实现  
// 作者两个功能一起实现了
template <typename Dtype>
inline void im2col_nd_core_cpu(const Dtype* data_input, const bool im2col,
    const int num_spatial_axes, const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, Dtype* data_output) 
{

    // 如果不是im2col则表明是col2im,也就是说data_output是需要输出的原始图像大小的数据  
    if (!im2col) 
    {
        int im_size = im_shape[0];

        for (int i = 0; i < num_spatial_axes; ++i) 
        {
            im_size *= im_shape[1 + i];
        }
        
        caffe_set(im_size, Dtype(0), data_output);
    }
    
    // 一个kernel大小的块有多大
    int kernel_size = 1; 

    for (int i = 0; i < num_spatial_axes; ++i) 
    {
        kernel_size *= kernel_shape[i];
    }
    
    // channels_col = inputchannel(输入图像的channel)*kernel_size  
    const int channels_col = col_shape[0]; 

    // 类似于im2col中的w_offset和h_offset，只不过因为这里是n维，所以用数组表示  
    vector<int> d_offset(num_spatial_axes, 0); 

    // 类似于im2col中w和h，是col_buff中的偏移  
    vector<int> d_iter(num_spatial_axes, 0); 

    for (int c_col = 0; c_col < channels_col; ++c_col) 
    {
        // Loop over spatial axes in reverse order to compute a per-axis offset.
        // Loop over spatial axes in reverse order to compute a per-axis offset.  
        // 计算n维kernel上的offset,与im2col中对应的代码一样的道理  
        // 只不过这里是n维了，所以用d_offset来表示  
        // 注意，这里用逆序来进行计算得到每个轴的偏移 
        int offset = c_col;
        
        for (int d_i = num_spatial_axes - 1; d_i >= 0; --d_i) 
        {
            if (d_i < num_spatial_axes - 1) 
            {
                offset /= kernel_shape[d_i + 1];
            }
            
            d_offset[d_i] = offset % kernel_shape[d_i];
        }
        
        for (bool incremented = true; incremented; ) 
        {
            // Loop over spatial axes in forward order to compute the indices in the
            // image and column, and whether the index lies in the padding.
            // 是经过im2colnd变换之后的索引  
            int index_col = c_col;

            // index_im是原始图像中的channel  
            int index_im = c_col / kernel_size;      
            bool is_padding = false;
            
            for (int d_i = 0; d_i < num_spatial_axes; ++d_i) 
            {
                // d是col_buff上的偏移，与d_pad相对(d_pad是原始图像上的偏移) 
                const int d = d_iter[d_i]; 

                // 在d_pad是经过pad之后的col_buff中的坐标经过转换成原图中的坐标  
                const int d_im = d * stride[d_i] - pad[d_i] +
                d_offset[d_i] * dilation[d_i];

                // 判断经过im2colnd处理的图像上的像素是否位于输入的n维图像的上的pad的那个部分
                is_padding |= d_im < 0 || d_im >= im_shape[d_i + 1]; 

                // 计算位于col_buff中的位置(就是经过im2colnd变换之后的) 
                index_col *= col_shape[d_i + 1]; 
                index_col += d;

                // 计算位于原始图像中的位置 
                index_im *= im_shape[d_i + 1];  
                index_im += d_im;
            }
            
            if (im2col) 
            {
                if (is_padding) 
                { 
                    // 如果是位于pad的部分则设置为0  
                    data_output[index_col] = 0;
                } 
                else 
                {
                    data_output[index_col] = data_input[index_im];
                }
            } 
            else if (!is_padding) 
            {  
                // col2im
                data_output[index_im] += data_input[index_col];
            }

            // 更新位于col_buff上的偏移d(d_iter就是所有的d存进去的)  
            // Loop over spatial axes in reverse order to choose an index,
            // like counting.
            incremented = false;
        
            for (int d_i = num_spatial_axes - 1; d_i >= 0; --d_i) 
            {
                const int d_max = col_shape[d_i + 1];
                DCHECK_LT(d_iter[d_i], d_max);
                
                if (d_iter[d_i] == d_max - 1) 
                {
                    d_iter[d_i] = 0;
                } 
                else 
                {  
                    // d_iter[d_i] < d_max - 1
                    ++d_iter[d_i];
                    incremented = true;
                    break;
                }
            }
        }  // while(incremented) {
    }  // for (int c = 0; c < channels_col; ++c) {
}

// 给出包裹im2col_nd_core_cpu 的im2col_nd_cpu 函数
// kIm2Col=true，输入是data_im，输出是data_col

// im2col_nd_cpu只是将kIm2Col=true然后调用im2col_nd_core_cpu 
template <typename Dtype>
void im2col_nd_cpu(const Dtype* data_im, const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, Dtype* data_col) {
  const bool kIm2Col = true;
  im2col_nd_core_cpu(data_im, kIm2Col, num_spatial_axes, im_shape, col_shape,
                  kernel_shape, pad, stride, dilation, data_col);
}

// Explicit instantiation
template void im2col_nd_cpu<float>(const float* data_im,
    const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, float* data_col);
template void im2col_nd_cpu<double>(const double* data_im,
    const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, double* data_col);

// 而对应的col2im的代码就很类似了与im2col 的代码几乎没有啥差别就是这个下面的赋值语句的位置颠倒了一下
template <typename Dtype>
void col2im_cpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    Dtype* data_im) {
  caffe_set(height * width * channels, Dtype(0), data_im);
  const int output_h = (height + 2 * pad_h -
    (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w -
    (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            data_col += output_w;
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                // 将处理后图像的每一个像素找到位于输入图像中的像素值
                data_im[input_row * width + input_col] += *data_col;
              }
              data_col++;
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

// Explicit instantiation
template void col2im_cpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    float* data_im);
template void col2im_cpu<double>(const double* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    double* data_im);

// 给出包裹im2col_nd_core_cpu的col2im_nd_cpu函数
// 一个德行，只不过kIm2Col = false了，此外输入的书data_col而输出的是data_im
template <typename Dtype>
void col2im_nd_cpu(const Dtype* data_col, const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, Dtype* data_im) {
  const bool kIm2Col = false;
  im2col_nd_core_cpu(data_col, kIm2Col, num_spatial_axes, im_shape, col_shape,
                     kernel_shape, pad, stride, dilation, data_im);
}

// Explicit instantiation
template void col2im_nd_cpu<float>(const float* data_col,
    const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, float* data_im);
template void col2im_nd_cpu<double>(const double* data_col,
    const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, double* data_im);


}  // namespace caffe
