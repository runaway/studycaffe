#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {  
  
// ��ʼ��DataReader�������  
template <typename Dtype>  
DataLayer<Dtype>::DataLayer(const LayerParameter& param)  
  : BasePrefetchingDataLayer<Dtype>(param),  
    reader_(param) {  
}  
  
// ��������ֹͣ�ڲ��߳�  
template <typename Dtype>  
DataLayer<Dtype>::~DataLayer() {  
  this->StopInternalThread();  
}  
  
// ���ݲ�ĳ�ʼ��  
template <typename Dtype>  
void DataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,  
      const vector<Blob<Dtype>*>& top) {  
  // �Ӳ�����ж�ȡbatch_size  
  const int batch_size = this->layer_param_.data_param().batch_size();  
  // Read a data point, and use it to initialize the top blob.  
  // ��reader_�л�ȡһ������  
  Datum& datum = *(reader_.full().peek());  
  
  // Use data_transformer to infer the expected blob shape from datum.  
  // ���������ƶ�blob����״��ŵ�top_shape  
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);  
  this->transformed_data_.Reshape(top_shape);  
  // Reshape top[0] and prefetch_data according to the batch_size.  
  // ��Ȼ��ȡ�����ݵ���״(channel,height,width)����ô����������һ��batch_size  
  // top_shape[0]=batch_size  
  // top_shape[1]=channel  
  // top_shape[2]=height  
  // top_shape[3]=width  
  top_shape[0] = batch_size;  
  // ������״����top[0]����״  
  top[0]->Reshape(top_shape);  
  
  // ����Ԥȡ���ݵ���״  
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {  
    this->prefetch_[i].data_.Reshape(top_shape);  
  }  
  LOG(INFO) << "output data size: " << top[0]->num() << ","  
      << top[0]->channels() << "," << top[0]->height() << ","  
      << top[0]->width();  
  // label  
  // ���������Ļ����top[1]����״ҲŪһ��  
  if (this->output_labels_) {  
    vector<int> label_shape(1, batch_size);  
    top[1]->Reshape(label_shape);  
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {  
      this->prefetch_[i].label_.Reshape(label_shape);  
    }  
  }  
}  
  
// This function is called on prefetch thread  
// ������������Լ�������߳�ִ�к����ڲ�ִ�е�  
template<typename Dtype>  
void DataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {  
  CPUTimer batch_timer;  
  batch_timer.Start();  
  double read_time = 0;  
  double trans_time = 0;  
  CPUTimer timer;  
  CHECK(batch->data_.count());  
  CHECK(this->transformed_data_.count());  
  
  // Reshape according to the first datum of each batch  
  // on single input batches allows for inputs of varying dimension.  
  // ��˼���������������������Ļ���ÿ��batch�����ݵ�ά�ȿ��Բ�һ��  
  // �Ӳ����ļ���ȡbatch_size  
  const int batch_size = this->layer_param_.data_param().batch_size();  
  // ��ȡ��һ������  
  Datum& datum = *(reader_.full().peek());  
  // Use data_transformer to infer the expected blob shape from datum.  
  // ʹ�õ�һ�������ƶ�blob����״  
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);  
  this->transformed_data_.Reshape(top_shape);  
  // Reshape batch according to the batch_size.  
  top_shape[0] = batch_size;  
  batch->data_.Reshape(top_shape);  
  
  // top_data������  
  Dtype* top_data = batch->data_.mutable_cpu_data();  
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables  
  
  // top_label�����  
  if (this->output_labels_) {  
    top_label = batch->label_.mutable_cpu_data();  
  }  
  
  // ���������ݽ��д���  
  for (int item_id = 0; item_id < batch_size; ++item_id) {  
    timer.Start();  
    // get a datum  
    Datum& datum = *(reader_.full().pop("Waiting for data"));  
    read_time += timer.MicroSeconds();  
    timer.Start();  
    // Apply data transformations (mirror, scale, crop...)  
    // ���ڸ����������ݻ�ȡoffset��������õ��Ǹ���batchid��Ȼ���ȡoffset  
    int offset = batch->data_.offset(item_id);  
    this->transformed_data_.set_cpu_data(top_data + offset);  
    this->data_transformer_->Transform(datum, &(this->transformed_data_));  
    // Copy label.  
    // �������  
    if (this->output_labels_) {  
      top_label[item_id] = datum.label();  
    }  
    // ���ݴ���ʱ��  
    trans_time += timer.MicroSeconds();  
  
    // ������ָ��ѹ��free����  
    reader_.free().push(const_cast<Datum*>(&datum));  
  }  
  timer.Stop();  
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(DataLayer);
REGISTER_LAYER_CLASS(Data);

}  // namespace caffe
