#include <vector>

#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe 
{
// �����������һ�������ͺ�����������0<=a<bΪ�棬����Ҳ��������һ��������������  
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

// ����im2col_cpu��ʱ������Ĳ���Ϊ
// im2col_cpu(һ��ͼ������ͼ���channel, ����ͼ���height, ����ͼ���width, kernel��height, kernel��width, pad��height, pad��width, stride��height�� stride��width)

// �������ͼ�����Ƚ������pad(ɶ�������䣬����ʵ��û��䣬����Ŀ��ͼ��������
// ����0)  
// �����һ����������ԭͼ���ϲ�û����pad,ֻ���ڴ�����ͼ���ϼ�����pad��ֵ  
// Ȼ����channel*kernel_h*kernel_wһ�У���һ��channel x kernel_h x kernel_w 
// ��С��ͼ�����һ���С�  
// �ж��ٸ����������أ���Ϳ����ù�ʽ���м���  
// ���� = [(ͼ��߶�+2*���߶�-kernel�߶�)/stride�߶�+1] * [(ͼ����+2*����
// ��-kernel���)/stride���+1]  
// �����������һ��kernel��С��ͼ����ά��  
// �������ʵ���Ͼ���kernel��ͼ���ϻ����Ĵ���
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

    // ����һ��kernel��С��ͼ��
    for (int channel = channels; channel--; data_im += channel_size) 
    {
        // ���������Ǽ�����kernel��С��ͼ�������λ��  
        //  c_im     h_offset     w_offset  

        // �������֮���ͼ��������ÿһ������  
        for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) 
        {
            for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) 
            {

                // ������֮���ͼ������֮ǰ��ͼ���λ��  
   
                // ���֮���ͼ������֮ǰ��ͼ����������Ӧ��λ��  
                // ���֮�������Ϊh��w��ô����Ӧ��ԭͼ���λ��Ϊ [h * stride_h - pad_h,   h * stride_h - pad_h+kernel_h]�Լ�  
                // [w * stride_w - pad_w,   w * stride_w - pad_w+kernel_w]  
                int input_row = -pad_h + kernel_row * dilation_h;

                for (int output_rows = output_h; output_rows; output_rows--) 
                {
                    // �������input_row>=height����ѭ������Ĵ���,Ȼ��ȥ������height�Ĳ���  
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
                            // �������input_col<width����ѭ������Ĵ���,Ȼ�����λ�õ�kernel�ĵ�ַ����data_col,����ȥ������width�Ĳ���  
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
��������˶�ά�������ô���Ǿͳ��ȴ������ٿ���nάͨ�þ�������ʵ�ֵ�
����������nάͨ�õľ���ľ���ʵ��
nά�����ʵ�����ά�����ʵ�ֺ����ƣ�ֻ������Ӧ�ı��������˱仯����ֻ��Ҫ�ҵ���
Ӧ�Ϳ��Ժܿ����

d_offset ��Ӧ��im2col�е�h_offset��w_offset��һ������ͼ���channel ����
kernel_size��С��ͼ����ƫ����(kernel_size����Ĵ����ж���)
d_iter��Ӧ��im2col���ڲ�forѭ����h��w���Ǿ���im2colnd�������col_buff�е�ƫ��
d_pad��Ӧ��im2col���ڲ�forѭ����h_pad��w_pad���������ԭʼͼ���е�ƫ��

���߻���im2colnd��col2imnd�ϲ���һ��ʵ���ˣ�ͨ��const bool im2col���ж���im2col
����col2im
*/

// nάͨ��im2col�Լ�col2im��ʵ��  
// ������������һ��ʵ����
template <typename Dtype>
inline void im2col_nd_core_cpu(const Dtype* data_input, const bool im2col,
    const int num_spatial_axes, const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, Dtype* data_output) 
{

    // �������im2col�������col2im,Ҳ����˵data_output����Ҫ�����ԭʼͼ���С������  
    if (!im2col) 
    {
        int im_size = im_shape[0];

        for (int i = 0; i < num_spatial_axes; ++i) 
        {
            im_size *= im_shape[1 + i];
        }
        
        caffe_set(im_size, Dtype(0), data_output);
    }
    
    // һ��kernel��С�Ŀ��ж��
    int kernel_size = 1; 

    for (int i = 0; i < num_spatial_axes; ++i) 
    {
        kernel_size *= kernel_shape[i];
    }
    
    // channels_col = inputchannel(����ͼ���channel)*kernel_size  
    const int channels_col = col_shape[0]; 

    // ������im2col�е�w_offset��h_offset��ֻ������Ϊ������nά�������������ʾ  
    vector<int> d_offset(num_spatial_axes, 0); 

    // ������im2col��w��h����col_buff�е�ƫ��  
    vector<int> d_iter(num_spatial_axes, 0); 

    for (int c_col = 0; c_col < channels_col; ++c_col) 
    {
        // Loop over spatial axes in reverse order to compute a per-axis offset.
        // Loop over spatial axes in reverse order to compute a per-axis offset.  
        // ����nάkernel�ϵ�offset,��im2col�ж�Ӧ�Ĵ���һ���ĵ���  
        // ֻ����������nά�ˣ�������d_offset����ʾ  
        // ע�⣬���������������м���õ�ÿ�����ƫ�� 
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
            // �Ǿ���im2colnd�任֮�������  
            int index_col = c_col;

            // index_im��ԭʼͼ���е�channel  
            int index_im = c_col / kernel_size;      
            bool is_padding = false;
            
            for (int d_i = 0; d_i < num_spatial_axes; ++d_i) 
            {
                // d��col_buff�ϵ�ƫ�ƣ���d_pad���(d_pad��ԭʼͼ���ϵ�ƫ��) 
                const int d = d_iter[d_i]; 

                // ��d_pad�Ǿ���pad֮���col_buff�е����꾭��ת����ԭͼ�е�����  
                const int d_im = d * stride[d_i] - pad[d_i] +
                d_offset[d_i] * dilation[d_i];

                // �жϾ���im2colnd�����ͼ���ϵ������Ƿ�λ�������nάͼ����ϵ�pad���Ǹ�����
                is_padding |= d_im < 0 || d_im >= im_shape[d_i + 1]; 

                // ����λ��col_buff�е�λ��(���Ǿ���im2colnd�任֮���) 
                index_col *= col_shape[d_i + 1]; 
                index_col += d;

                // ����λ��ԭʼͼ���е�λ�� 
                index_im *= im_shape[d_i + 1];  
                index_im += d_im;
            }
            
            if (im2col) 
            {
                if (is_padding) 
                { 
                    // �����λ��pad�Ĳ���������Ϊ0  
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

            // ����λ��col_buff�ϵ�ƫ��d(d_iter�������е�d���ȥ��)  
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

// ��������im2col_nd_core_cpu ��im2col_nd_cpu ����
// kIm2Col=true��������data_im�������data_col

// im2col_nd_cpuֻ�ǽ�kIm2Col=trueȻ�����im2col_nd_core_cpu 
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

// ����Ӧ��col2im�Ĵ���ͺ���������im2col �Ĵ��뼸��û��ɶ�������������ĸ�ֵ����λ�õߵ���һ��
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
                // �������ͼ���ÿһ�������ҵ�λ������ͼ���е�����ֵ
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

// ��������im2col_nd_core_cpu��col2im_nd_cpu����
// һ�����У�ֻ����kIm2Col = false�ˣ������������data_col���������data_im
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
