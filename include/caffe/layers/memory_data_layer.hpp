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
（8）MemoryDataLayer 类的定义以及实现如下：
该类主要就是对于读取好的Datum或者OpenCV读取的Mat的Vector进行预处理（图像的crop、scale等），然后前传。

首先给出该类的定义

*/
/** 
 * @brief Provides data to the Net from memory. 
 * 从内存中读取数据，这里指已经从数据文件或者图像文件中读取到了数据，然后输入到该层 
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
  
  // 将内存中的数据加入added_data_和added_label_(数据和类标)  
  virtual void AddDatumVector(const vector<Datum>& datum_vector);  
#ifdef USE_OPENCV  
  // 如果有opencv则将opencv读取到的Mat,并且将labels加入added_data_和added_label_(数据和类标)  
  virtual void AddMatVector(const vector<cv::Mat>& mat_vector,  
      const vector<int>& labels);  
#endif  // USE_OPENCV  
  
  // Reset should accept const pointers, but can't, because the memory  
  //  will be given to Blob, which is mutable  
  // Reset函数实际上是将data、label、以及batchsize(n)设置到内部的变量里面去  
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
  // 内部的数据和类标  
  Blob<Dtype> added_data_;  
  Blob<Dtype> added_label_;  
  // 是否有新的数据  
  bool has_new_data_;  
};  

}  // namespace caffe

#endif  // CAFFE_MEMORY_DATA_LAYER_HPP_
