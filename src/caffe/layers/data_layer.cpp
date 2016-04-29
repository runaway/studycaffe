#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {  
  
// 初始化DataReader，层参数  
template <typename Dtype>  
DataLayer<Dtype>::DataLayer(const LayerParameter& param)  
  : BasePrefetchingDataLayer<Dtype>(param),  
    reader_(param) {  
}  
  
// 析构函数停止内部线程  
template <typename Dtype>  
DataLayer<Dtype>::~DataLayer() {  
  this->StopInternalThread();  
}  
  
// 数据层的初始化  
template <typename Dtype>  
void DataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,  
      const vector<Blob<Dtype>*>& top) {  
  // 从层参数中读取batch_size  
  const int batch_size = this->layer_param_.data_param().batch_size();  
  // Read a data point, and use it to initialize the top blob.  
  // 从reader_中获取一个数据  
  Datum& datum = *(reader_.full().peek());  
  
  // Use data_transformer to infer the expected blob shape from datum.  
  // 用数据来推断blob的形状存放到top_shape  
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);  
  this->transformed_data_.Reshape(top_shape);  
  // Reshape top[0] and prefetch_data according to the batch_size.  
  // 既然获取了数据的形状(channel,height,width)，那么这里再设置一下batch_size  
  // top_shape[0]=batch_size  
  // top_shape[1]=channel  
  // top_shape[2]=height  
  // top_shape[3]=width  
  top_shape[0] = batch_size;  
  // 根据形状设置top[0]的形状  
  top[0]->Reshape(top_shape);  
  
  // 设置预取数据的形状  
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {  
    this->prefetch_[i].data_.Reshape(top_shape);  
  }  
  LOG(INFO) << "output data size: " << top[0]->num() << ","  
      << top[0]->channels() << "," << top[0]->height() << ","  
      << top[0]->width();  
  // label  
  // 如果输出类标的话则把top[1]的形状也弄一下  
  if (this->output_labels_) {  
    vector<int> label_shape(1, batch_size);  
    top[1]->Reshape(label_shape);  
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {  
      this->prefetch_[i].label_.Reshape(label_shape);  
    }  
  }  
}  
  
// This function is called on prefetch thread  
// 这个函数是在自己定义的线程执行函数内部执行的  
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
  // 意思是像以下这种做法这样的话，每个batch的数据的维度可以不一样  
  // 从参数文件获取batch_size  
  const int batch_size = this->layer_param_.data_param().batch_size();  
  // 获取第一个数据  
  Datum& datum = *(reader_.full().peek());  
  // Use data_transformer to infer the expected blob shape from datum.  
  // 使用第一个数据推断blob的形状  
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);  
  this->transformed_data_.Reshape(top_shape);  
  // Reshape batch according to the batch_size.  
  top_shape[0] = batch_size;  
  batch->data_.Reshape(top_shape);  
  
  // top_data存数据  
  Dtype* top_data = batch->data_.mutable_cpu_data();  
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables  
  
  // top_label存类标  
  if (this->output_labels_) {  
    top_label = batch->label_.mutable_cpu_data();  
  }  
  
  // 对这批数据进行处理  
  for (int item_id = 0; item_id < batch_size; ++item_id) {  
    timer.Start();  
    // get a datum  
    Datum& datum = *(reader_.full().pop("Waiting for data"));  
    read_time += timer.MicroSeconds();  
    timer.Start();  
    // Apply data transformations (mirror, scale, crop...)  
    // 对于给定批的数据获取offset，这里调用的是给定batchid，然后获取offset  
    int offset = batch->data_.offset(item_id);  
    this->transformed_data_.set_cpu_data(top_data + offset);  
    this->data_transformer_->Transform(datum, &(this->transformed_data_));  
    // Copy label.  
    // 复制类标  
    if (this->output_labels_) {  
      top_label[item_id] = datum.label();  
    }  
    // 数据传输时间  
    trans_time += timer.MicroSeconds();  
  
    // 将数据指针压到free队列  
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
