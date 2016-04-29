#ifndef CAFFE_MEMORY_DATA_LAYER_HPP_
#define CAFFE_MEMORY_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

/**
 * @brief Provides data to the Net from memory.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
/*
��8��MemoryDataLayer ��Ķ����Լ�ʵ�����£�
������Ҫ���Ƕ��ڶ�ȡ�õ�Datum����OpenCV��ȡ��Mat��Vector����Ԥ����ͼ���crop��scale�ȣ���Ȼ��ǰ����

���ȸ�������Ķ���

*/
/** 
 * @brief Provides data to the Net from memory. 
 * ���ڴ��ж�ȡ���ݣ�����ָ�Ѿ��������ļ�����ͼ���ļ��ж�ȡ�������ݣ�Ȼ�����뵽�ò� 
 * TODO(dox): thorough documentation for Forward and proto params. 
 */  
template <typename Dtype>  
class MemoryDataLayer : public BaseDataLayer<Dtype> {  
 public:  
  explicit MemoryDataLayer(const LayerParameter& param)  
      : BaseDataLayer<Dtype>(param), has_new_data_(false) {}  
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,  
      const vector<Blob<Dtype>*>& top);  
  
  virtual inline const char* type() const { return "MemoryData"; }  
  virtual inline int ExactNumBottomBlobs() const { return 0; }  
  virtual inline int ExactNumTopBlobs() const { return 2; }  
  
  // ���ڴ��е����ݼ���added_data_��added_label_(���ݺ����)  
  virtual void AddDatumVector(const vector<Datum>& datum_vector);  
#ifdef USE_OPENCV  
  // �����opencv��opencv��ȡ����Mat,���ҽ�labels����added_data_��added_label_(���ݺ����)  
  virtual void AddMatVector(const vector<cv::Mat>& mat_vector,  
      const vector<int>& labels);  
#endif  // USE_OPENCV  
  
  // Reset should accept const pointers, but can't, because the memory  
  //  will be given to Blob, which is mutable  
  // Reset����ʵ�����ǽ�data��label���Լ�batchsize(n)���õ��ڲ��ı�������ȥ  
  void Reset(Dtype* data, Dtype* label, int n);  
  void set_batch_size(int new_size);  
  
  int batch_size() { return batch_size_; }  
  int channels() { return channels_; }  
  int height() { return height_; }  
  int width() { return width_; }  
  
 protected:  
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,  
      const vector<Blob<Dtype>*>& top);  
  
  int batch_size_, channels_, height_, width_, size_;  
  Dtype* data_;  
  Dtype* labels_;  
  // batch_size  
  int n_;  
  size_t pos_;  
  // �ڲ������ݺ����  
  Blob<Dtype> added_data_;  
  Blob<Dtype> added_label_;  
  // �Ƿ����µ�����  
  bool has_new_data_;  
};  

}  // namespace caffe

#endif  // CAFFE_MEMORY_DATA_LAYER_HPP_
