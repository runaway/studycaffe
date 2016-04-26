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
һ��Data_layers.hpp�ļ������ü��

Data_layers.hpp��Ŀǰcaffe��master��֧���Ѿ����ܴ����ˣ���ɢ�������ļ���ȥ�ˡ�
��֮ǰ�Ǵ�����cafferoot\include\caffe�С������Ѿ�����˸���������Ƶ�ͷ�ļ��ˡ�������������

���ȸ�������ļ����������ļ��������ݶ�ȡ�йص��ࡣ
�ֱ�Ϊ��
BaseDataLayer
���ݲ�Ļ��࣬�̳���ͨ�õ���Layer

Batch
Batchʵ���Ͼ���һ��data_��label_���

BasePrefetchingDataLayer
��Ԥȡ��Ļ��࣬�̳���BaseDataLayer��InternalThread�������ܹ���ȡһ�����ݵ�����

DataLayer
DataLayer�������ǣ��̳���BasePrefetchingDataLayer
ʹ��DataReader���������ݹ����Ӷ�ʵ�ֲ��л�

DummyDataLayer
�����Ǽ̳���Layer,ͨ��Filler��������

HDF5DataLayer
��HDF5�ж�ȡ���̳���Layer

HDF5OutputLayer
������д�뵽HDF5�ļ����̳���Layer

ImageDataLayer
��ͼ���ļ��ж�ȡ���ݣ����Ӧ�ñȽϳ��ã��̳���BasePrefetchingDataLayer

MemoryDataLayer
���ڴ��ж�ȡ���ݣ�����ָ�Ѿ��������ļ�����ͼ���ļ��ж�ȡ�������ݣ�Ȼ�����뵽�ò㣬�̳���BaseDataLayer


WindowDataLayer
��ͼ���ļ��Ĵ��ڻ�ȡ���ݣ���Ҫָ�����������ļ����̳���BasePrefetchingDataLayer

����Data_layers�ļ��ĵ���ϸ����
��������Ȼ��ͬһ��ͷ�ļ��н��еĶ��壬����ȴ�����ڲ�ͬ��cpp�ļ����е�ʵ�֡�
����������ʵ���ļ�
BaseDataLayer��BasePrefetchingDataLayer
��Ӧ�ڣ�
base_data_layer.cpp
base_data_layer.cu

DataLayer
��Ӧ�ڣ�
data_layer.cpp

DummyDataLayer
��Ӧ�ڣ�
dummy_data_layer.cpp


HDF5DataLayer
HDF5OutputLayer
��Ӧ�ڣ�
hdf5_data_layer.cpp
hdf5_data_layer.cu
�Լ�
hdf5_output_layer.cpp
hdf5_output_layer.cu

ImageDataLayer
��Ӧ�ڣ�
image_data_layer.cpp


MemoryDataLayer
��Ӧ�ڣ�
memory_data_layer.cpp


WindowDataLayer
��Ӧ��
window_data_layer.cpp

����������Щ�������ϸ������


��1��BaseDataLayer���ඨ���Լ�ʵ�����£�
*/
namespace caffe {

/** 
 * @brief Provides base for data layers that feed blobs to the Net. 
 * 
 * TODO(dox): thorough documentation for Forward and proto params. 
 * ���ݲ�Ļ��� 
 */  
template <typename Dtype>  
class BaseDataLayer : public Layer<Dtype> {  
 public:  
  // ��ʽ���캯��  
  explicit BaseDataLayer(const LayerParameter& param);  
  // LayerSetUp: implements common data layer setup functionality, and calls  
  // DataLayerSetUp to do special data layer setup for individual layer types.  
  // This method may not be overridden except by the BasePrefetchingDataLayer.  
  // �ú���ֻ�ܱ�BasePrefetchingDataLayer���������  
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,  
      const vector<Blob<Dtype>*>& top);  
  // Data layers should be shared by multiple solvers in parallel  
  // �����Ƿ���Ҫ���������solver���й���  
  virtual inline bool ShareInParallel() const { return true; }  
  
  // ���ݲ�ĳ�ʼ��  
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,  
      const vector<Blob<Dtype>*>& top) {}  
  
  // ���ݲ���û�������(��bottoms)������reshapeֻ����ʽ  
  // Data layers have no bottoms, so reshaping is trivial.  
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,  
      const vector<Blob<Dtype>*>& top) {}  
  
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,  
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}  
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,  
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}  
  
 protected:  
  // ����������ݽ��б任�Ĳ����������а����Ƿ���Ҫmirror���Ƿ���Ҫcrop  
  // �Ƿ���Ҫ��ȥmeanfile���Ƿ���Ҫscale  
  TransformationParameter transform_param_;  
  // ʵ��ִ�����ݱ任���ָ��(һ��Transform�������ϲ���������ɶ����ݵı任�����������ݹ�)  
  shared_ptr<DataTransformer<Dtype> > data_transformer_;  
  bool output_labels_;  
};  

template <typename Dtype>
class Batch {
 public:
  Blob<Dtype> data_, label_;
};

// BasePrefetchingDataLayer���Ǽ̳���BaseDataLayer��  
// ��Ԥȡ��Ļ���  
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
  // ����load_batch�������ú����Ǵ��麯�����̳иú������඼��Ҫʵ�ֵ�  
  virtual void load_batch(Batch<Dtype>* batch) = 0;  
  // ����prefetch����,prefetch_free_,prefetch_full_  
  Batch<Dtype> prefetch_[PREFETCH_COUNT];  
  BlockingQueue<Batch<Dtype>*> prefetch_free_;  
  BlockingQueue<Batch<Dtype>*> prefetch_full_;  
  
  Blob<Dtype> transformed_data_;  
};  
  

}  // namespace caffe

#endif  // CAFFE_DATA_LAYERS_HPP_
