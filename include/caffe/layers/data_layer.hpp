#ifndef CAFFE_DATA_LAYER_HPP_
#define CAFFE_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

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


三、总结
首先理顺类与类之间的关系：
Layer类是所有神经网络层的基类，BaseDataLayer继承自该类，BasePrefetchingDataLayer继承自BaseDataLayer，DataLayer继承自BasePrefetchingDataLayer。
有了上述几个基础的类之后，其他的类都是从这几个类进行派生。

比如DummyDataLayer，HDF5Layer和HDF5OutputLayer都是直接继承自Layer。
MemoryDataLayer则是继承自BaseDataLayer

凡是涉及到直接读取数据文件的一般都是继承自BasePrefetchingDataLayer，这样可以有效地读数据进行预取。
比如：ImageDataLayer、WindowDataLayer
继承自BasePrefetchingDataLayer需要实现load_batch函数以供内部的线程进行调用，实现数据预取。
此外每一个网络层的类（因为所有的网络层都继承自Layer类嘛）都需要实现SetUp，这个是必须的。

这一次的量还真有点大。。。
*/

/*
看到hdf5、leveldb、lmdb，确实是与具体数据相关了。data_layer作为原始数据的输入层，处于整个网络的最底层，它可以从数据库leveldb、lmdb中读取数据，也可以直接从内存中读取，还可以从hdf5，甚至是原始的图像读入数据。

关于这几个数据库，简介如下：

LevelDB是Google公司搞的一个高性能的key/value存储库，调用简单，数据是被Snappy压缩，据说效率很多，可以减少磁盘I/O，具体例子可以看看维基百科。

而LMDB（Lightning Memory-Mapped Database），是个和levelDB类似的key/value存储库，但效果似乎更好些，其首页上写道“ultra-fast，ultra-compact”，这个有待进一步学习啊～～

HDF（Hierarchical Data Format）是一种为存储和处理大容量科学数据而设计的文件格式及相应的库文件，当前最流行的版本是HDF5,其文件包含两种基本数据对象：

群组（group）：类似文件夹，可以包含多个数据集或下级群组；
数据集（dataset）：数据内容，可以是多维数组，也可以是更复杂的数据类型。
*/

namespace caffe {

template <typename Dtype>
class DataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit DataLayer(const LayerParameter& param);
  virtual ~DataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // DataLayer uses DataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "Data"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void load_batch(Batch<Dtype>* batch);

  DataReader reader_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYER_HPP_
