#ifndef CAFFE_IM2COL_LAYER_HPP_
#define CAFFE_IM2COL_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief A helper for image operations that rearranges image regions into
 *        column vectors.  Used by ConvolutionLayer to perform convolution
 *        by matrix multiplication.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class Im2colLayer : public Layer<Dtype> {
 public:
  explicit Im2colLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Im2col"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

/// @brief The spatial dimensions of a filter kernel.  
  // kernel����״ = [kernel_h, kernel_w]  
  Blob<int> kernel_shape_;  
  
  /// @brief The spatial dimensions of the stride.  
  // ������״ = [stride_h, stride_w]  
  Blob<int> stride_;  
  
  /// @brief The spatial dimensions of the padding.  
  // pad����״ = [pad_h, pad_w]  
  Blob<int> pad_;
  /// @brief The spatial dimensions of the dilation.
  Blob<int> dilation_;

  // �ռ������  
  int num_spatial_axes_;  
  
  // �����ά�� = ����ͼ��ͨ����*����ͼ���h*����ͼ��w  
  int bottom_dim_;  
  
  // ���ά�� = ���ͨ����*���h*���w  
  int top_dim_;  
  
  // ����ͼ��ĵڼ�������ͨ��  
  int channel_axis_;  
  
  // batchsize  
  int num_;  
  
  // ����ͼ���ͨ����  
  int channels_;  

  bool force_nd_im2col_;
};

}  // namespace caffe

#endif  // CAFFE_IM2COL_LAYER_HPP_
