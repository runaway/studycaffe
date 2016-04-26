#ifndef CAFFE_DATA_LAYERS_HPP_
#define CAFFE_DATA_LAYERS_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"
/*
一、Data_layers.hpp文件的作用简介

Data_layers.hpp在目前caffe的master分支中已经不能存在了，分散到各个文件中去了。
而之前是存在于cafferoot\include\caffe中。现在已经变成了各个类的名称的头文件了。这里做个提醒

首先给出这个文件中所包含的几个与数据读取有关的类。
分别为：
BaseDataLayer
数据层的基类，继承自通用的类Layer

Batch
Batch实际上就是一个data_和label_类标

BasePrefetchingDataLayer
是预取层的基类，继承自BaseDataLayer和InternalThread，包含能够读取一批数据的能力

DataLayer
DataLayer才是主角，继承自BasePrefetchingDataLayer
使用DataReader来进行数据共享，从而实现并行化

DummyDataLayer
该类是继承自Layer,通过Filler产生数据

HDF5DataLayer
从HDF5中读取，继承自Layer

HDF5OutputLayer
将数据写入到HDF5文件，继承自Layer

ImageDataLayer
从图像文件中读取数据，这个应该比较常用，继承自BasePrefetchingDataLayer

MemoryDataLayer
从内存中读取数据，这里指已经从数据文件或者图像文件中读取到了数据，然后输入到该层，继承自BaseDataLayer


WindowDataLayer
从图像文件的窗口获取数据，需要指定窗口数据文件，继承自BasePrefetchingDataLayer

二、Data_layers文件的的详细介绍
上述类虽然在同一个头文件中进行的定义，但是却都是在不同的cpp文件进行的实现。
下面给出类的实现文件
BaseDataLayer和BasePrefetchingDataLayer
对应于：
base_data_layer.cpp
base_data_layer.cu

DataLayer
对应于：
data_layer.cpp

DummyDataLayer
对应于：
dummy_data_layer.cpp


HDF5DataLayer
HDF5OutputLayer
对应于：
hdf5_data_layer.cpp
hdf5_data_layer.cu
以及
hdf5_output_layer.cpp
hdf5_output_layer.cu

ImageDataLayer
对应于：
image_data_layer.cpp


MemoryDataLayer
对应于：
memory_data_layer.cpp


WindowDataLayer
对应于
window_data_layer.cpp

接下来对这些类进行详细阐述：


（1）BaseDataLayer的类定义以及实现如下：
*/
namespace caffe {

/** 
 * @brief Provides base for data layers that feed blobs to the Net. 
 * 
 * TODO(dox): thorough documentation for Forward and proto params. 
 * 数据层的基类 
 */  
template <typename Dtype>  
class BaseDataLayer : public Layer<Dtype> {  
 public:  
  // 显式构造函数  
  explicit BaseDataLayer(const LayerParameter& param);  
  // LayerSetUp: implements common data layer setup functionality, and calls  
  // DataLayerSetUp to do special data layer setup for individual layer types.  
  // This method may not be overridden except by the BasePrefetchingDataLayer.  
  // 该函数只能被BasePrefetchingDataLayer层进行重载  
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,  
      const vector<Blob<Dtype>*>& top);  
  // Data layers should be shared by multiple solvers in parallel  
  // 数据是否需要给多个并行solver进行共享  
  virtual inline bool ShareInParallel() const { return true; }  
  
  // 数据层的初始化  
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,  
      const vector<Blob<Dtype>*>& top) {}  
  
  // 数据层是没有输入的(即bottoms)，所以reshape只是形式  
  // Data layers have no bottoms, so reshaping is trivial.  
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,  
      const vector<Blob<Dtype>*>& top) {}  
  
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,  
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}  
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,  
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}  
  
 protected:  
  // 对输入的数据进行变换的参数，这其中包括是否需要mirror，是否需要crop  
  // 是否需要减去meanfile，是否需要scale  
  TransformationParameter transform_param_;  
  // 实际执行数据变换类的指针(一个Transform函数加上参数即可完成对数据的变换，参数是数据哈)  
  shared_ptr<DataTransformer<Dtype> > data_transformer_;  
  bool output_labels_;  
};  

template <typename Dtype>
class Batch {
 public:
  Blob<Dtype> data_, label_;
};

// BasePrefetchingDataLayer层是继承于BaseDataLayer的  
// 是预取层的基类  
template <typename Dtype>  
class BasePrefetchingDataLayer :  
    public BaseDataLayer<Dtype>, public InternalThread {  
 public:  
  explicit BasePrefetchingDataLayer(const LayerParameter& param);  
  // LayerSetUp: implements common data layer setup functionality, and calls  
  // DataLayerSetUp to do special data layer setup for individual layer types.  
  // This method may not be overridden.  
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,  
      const vector<Blob<Dtype>*>& top);  
  
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,  
      const vector<Blob<Dtype>*>& top);  
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,  
      const vector<Blob<Dtype>*>& top);  
  
  // Prefetches batches (asynchronously if to GPU memory)  
  static const int PREFETCH_COUNT = 3;  
  
 protected:  
  virtual void InternalThreadEntry();  
  // 多了load_batch函数，该函数是纯虚函数，继承该函数的类都需要实现的  
  virtual void load_batch(Batch<Dtype>* batch) = 0;  
  // 还有prefetch数组,prefetch_free_,prefetch_full_  
  Batch<Dtype> prefetch_[PREFETCH_COUNT];  
  BlockingQueue<Batch<Dtype>*> prefetch_free_;  
  BlockingQueue<Batch<Dtype>*> prefetch_full_;  
  
  Blob<Dtype> transformed_data_;  
};  
  

}  // namespace caffe

#endif  // CAFFE_DATA_LAYERS_HPP_
