#include <boost/thread.hpp>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

// 构造函数就是初始化数据变换参数  
template <typename Dtype>  
BaseDataLayer<Dtype>::BaseDataLayer(const LayerParameter& param)  
    : Layer<Dtype>(param),  
      transform_param_(param.transform_param()) {  
}  
  
// 初始化的时候根据top的大小来确定，如果是1表明只输出数据，而不输出类标  
template <typename Dtype>  
void BaseDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,  
      const vector<Blob<Dtype>*>& top) {  
  if (top.size() == 1) {  
    output_labels_ = false;  
  } else {  
    output_labels_ = true;  
  }  
  // 初始化一个DataTransformer实例，便于对数据进行预处理  
  data_transformer_.reset(  
      new DataTransformer<Dtype>(transform_param_, this->phase_));  
  // 初始化种子  
  data_transformer_->InitRand();  
  // The subclasses should setup the size of bottom and top  
  // 执行数据层的初始化  
  DataLayerSetUp(bottom, top);  
}  

template <typename Dtype>
BasePrefetchingDataLayer<Dtype>::BasePrefetchingDataLayer(
    const LayerParameter& param)
    : BaseDataLayer<Dtype>(param),
      prefetch_free_(), prefetch_full_() {
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
    prefetch_free_.push(&prefetch_[i]);
  }
}

// 进行层的初始化  
template <typename Dtype>  
void BasePrefetchingDataLayer<Dtype>::LayerSetUp(  
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {  
    // 首先执行基类BaseDataLayer的层初始化  
  BaseDataLayer<Dtype>::LayerSetUp(bottom, top);  
  // Before starting the prefetch thread, we make cpu_data and gpu_data  
  // calls so that the prefetch thread does not accidentally make simultaneous  
  // cudaMalloc calls when the main thread is running. In some GPUs this  
  // seems to cause failures if we do not so.  
  // 在开启预取线程的时候，需要让cpu数据和gpu数据分配空间  
  // 这样才能够避免在某些GPU上出现问题  
  
  // 首先是CPU  
  for (int i = 0; i < PREFETCH_COUNT; ++i) {  
    prefetch_[i].data_.mutable_cpu_data();  
    if (this->output_labels_) {  
      prefetch_[i].label_.mutable_cpu_data();  
    }  
  }  
#ifndef CPU_ONLY  
  // 然后是GPU  
  if (Caffe::mode() == Caffe::GPU) {  
    for (int i = 0; i < PREFETCH_COUNT; ++i) {  
      prefetch_[i].data_.mutable_gpu_data();  
      if (this->output_labels_) {  
        prefetch_[i].label_.mutable_gpu_data();  
      }  
    }  
  }  
#endif  
  DLOG(INFO) << "Initializing prefetch";  
  // 初始化随机数种子  
  this->data_transformer_->InitRand();  
  // 开启线程  
  StartInternalThread();  
  DLOG(INFO) << "Prefetch initialized.";  
}  
  
// 在StartInternalThread开启线程后就会执行下面自己定义的函数  
// 这个就是自己定义的函数，让线程去执行的  
template <typename Dtype>  
void BasePrefetchingDataLayer<Dtype>::InternalThreadEntry() {  
#ifndef CPU_ONLY  
  cudaStream_t stream;  
  if (Caffe::mode() == Caffe::GPU) {  
      // 创建非阻塞流  
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));  
  }  
#endif  
  
  try {  
    while (!must_stop()) {  
        // 弹出一个batch  
      Batch<Dtype>* batch = prefetch_free_.pop();  
        // 装载batch  
      load_batch(batch);  
#ifndef CPU_ONLY  
      if (Caffe::mode() == Caffe::GPU) {  
          // 如果GPU模式开始，则推送到GPU  
        batch->data_.data().get()->async_gpu_push(stream);  
        // 检查是否成功  
        CUDA_CHECK(cudaStreamSynchronize(stream));  
      }  
#endif  
      // 将装好的batch压入full队列  
      prefetch_full_.push(batch);  
    }  
  } catch (boost::thread_interrupted&) {  
    // Interrupted exception is expected on shutdown  
  }  
#ifndef CPU_ONLY  
  if (Caffe::mode() == Caffe::GPU) {  
      // 销毁流  
    CUDA_CHECK(cudaStreamDestroy(stream));  
  }  
#endif  
}  
  
template <typename Dtype>  
void BasePrefetchingDataLayer<Dtype>::Forward_cpu(  
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {  
    // 传递的时候是从full队列中弹出一个数据  
  Batch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");  
  // Reshape to loaded data.  
  // 根据batch的形状改变数据形状  
  top[0]->ReshapeLike(batch->data_);  
  // Copy the data  
  // 将batch数据复制到top[0]  
  caffe_copy(batch->data_.count(), batch->data_.cpu_data(),  
             top[0]->mutable_cpu_data());  
  DLOG(INFO) << "Prefetch copied";  
  if (this->output_labels_) {  
      // 输出类标的话  
    // Reshape to loaded labels.  
    // 根据batch中类标的形状改变top[1]的形状  
    top[1]->ReshapeLike(batch->label_);  
    // Copy the labels.  
    // 复制类标到top[1]  
    caffe_copy(batch->label_.count(), batch->label_.cpu_data(),  
        top[1]->mutable_cpu_data());  
  }  
  // 将该batch压入free队列  
  prefetch_free_.push(batch);  
}  
  
  
// 如果没有GPU的话则在BasePrefetchingDataLayer类中生成一个Forward函数  
// 该函数并不前传，而是直接报错  
#ifdef CPU_ONLY  
STUB_GPU_FORWARD(BasePrefetchingDataLayer, Forward);  
#endif  
// 初始化层  
INSTANTIATE_CLASS(BaseDataLayer);
INSTANTIATE_CLASS(BasePrefetchingDataLayer);

}  // namespace caffe
