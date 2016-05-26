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


�����ܽ�
������˳������֮��Ĺ�ϵ��
Layer���������������Ļ��࣬BaseDataLayer�̳��Ը��࣬BasePrefetchingDataLayer�̳���BaseDataLayer��DataLayer�̳���BasePrefetchingDataLayer��
��������������������֮���������඼�Ǵ��⼸�������������

����DummyDataLayer��HDF5Layer��HDF5OutputLayer����ֱ�Ӽ̳���Layer��
MemoryDataLayer���Ǽ̳���BaseDataLayer

�����漰��ֱ�Ӷ�ȡ�����ļ���һ�㶼�Ǽ̳���BasePrefetchingDataLayer������������Ч�ض����ݽ���Ԥȡ��
���磺ImageDataLayer��WindowDataLayer
�̳���BasePrefetchingDataLayer��Ҫʵ��load_batch�����Թ��ڲ����߳̽��е��ã�ʵ������Ԥȡ��
����ÿһ���������ࣨ��Ϊ���е�����㶼�̳���Layer�������Ҫʵ��SetUp������Ǳ���ġ�

��һ�ε��������е�󡣡���
*/

/*
����hdf5��leveldb��lmdb��ȷʵ���������������ˡ�data_layer��Ϊԭʼ���ݵ�����㣬���������������ײ㣬�����Դ����ݿ�leveldb��lmdb�ж�ȡ���ݣ�Ҳ����ֱ�Ӵ��ڴ��ж�ȡ�������Դ�hdf5��������ԭʼ��ͼ��������ݡ�

�����⼸�����ݿ⣬������£�

LevelDB��Google��˾���һ�������ܵ�key/value�洢�⣬���ü򵥣������Ǳ�Snappyѹ������˵Ч�ʺܶ࣬���Լ��ٴ���I/O���������ӿ��Կ���ά���ٿơ�

��LMDB��Lightning Memory-Mapped Database�����Ǹ���levelDB���Ƶ�key/value�洢�⣬��Ч���ƺ�����Щ������ҳ��д����ultra-fast��ultra-compact��������д���һ��ѧϰ������

HDF��Hierarchical Data Format����һ��Ϊ�洢�ʹ����������ѧ���ݶ���Ƶ��ļ���ʽ����Ӧ�Ŀ��ļ�����ǰ�����еİ汾��HDF5,���ļ��������ֻ������ݶ���

Ⱥ�飨group���������ļ��У����԰���������ݼ����¼�Ⱥ�飻
���ݼ���dataset�����������ݣ������Ƕ�ά���飬Ҳ�����Ǹ����ӵ��������͡�
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
