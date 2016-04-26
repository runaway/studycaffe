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

// ���캯�����ǳ�ʼ�����ݱ任����  
template <typename Dtype>  
BaseDataLayer<Dtype>::BaseDataLayer(const LayerParameter& param)  
    : Layer<Dtype>(param),  
      transform_param_(param.transform_param()) {  
}  
  
// ��ʼ����ʱ�����top�Ĵ�С��ȷ���������1����ֻ������ݣ�����������  
template <typename Dtype>  
void BaseDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,  
      const vector<Blob<Dtype>*>& top) {  
  if (top.size() == 1) {  
    output_labels_ = false;  
  } else {  
    output_labels_ = true;  
  }  
  // ��ʼ��һ��DataTransformerʵ�������ڶ����ݽ���Ԥ����  
  data_transformer_.reset(  
      new DataTransformer<Dtype>(transform_param_, this->phase_));  
  // ��ʼ������  
  data_transformer_->InitRand();  
  // The subclasses should setup the size of bottom and top  
  // ִ�����ݲ�ĳ�ʼ��  
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

// ���в�ĳ�ʼ��  
template <typename Dtype>  
void BasePrefetchingDataLayer<Dtype>::LayerSetUp(  
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {  
    // ����ִ�л���BaseDataLayer�Ĳ��ʼ��  
  BaseDataLayer<Dtype>::LayerSetUp(bottom, top);  
  // Before starting the prefetch thread, we make cpu_data and gpu_data  
  // calls so that the prefetch thread does not accidentally make simultaneous  
  // cudaMalloc calls when the main thread is running. In some GPUs this  
  // seems to cause failures if we do not so.  
  // �ڿ���Ԥȡ�̵߳�ʱ����Ҫ��cpu���ݺ�gpu���ݷ���ռ�  
  // �������ܹ�������ĳЩGPU�ϳ�������  
  
  // ������CPU  
  for (int i = 0; i < PREFETCH_COUNT; ++i) {  
    prefetch_[i].data_.mutable_cpu_data();  
    if (this->output_labels_) {  
      prefetch_[i].label_.mutable_cpu_data();  
    }  
  }  
#ifndef CPU_ONLY  
  // Ȼ����GPU  
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
  // ��ʼ�����������  
  this->data_transformer_->InitRand();  
  // �����߳�  
  StartInternalThread();  
  DLOG(INFO) << "Prefetch initialized.";  
}  
  
// ��StartInternalThread�����̺߳�ͻ�ִ�������Լ�����ĺ���  
// ��������Լ�����ĺ��������߳�ȥִ�е�  
template <typename Dtype>  
void BasePrefetchingDataLayer<Dtype>::InternalThreadEntry() {  
#ifndef CPU_ONLY  
  cudaStream_t stream;  
  if (Caffe::mode() == Caffe::GPU) {  
      // ������������  
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));  
  }  
#endif  
  
  try {  
    while (!must_stop()) {  
        // ����һ��batch  
      Batch<Dtype>* batch = prefetch_free_.pop();  
        // װ��batch  
      load_batch(batch);  
#ifndef CPU_ONLY  
      if (Caffe::mode() == Caffe::GPU) {  
          // ���GPUģʽ��ʼ�������͵�GPU  
        batch->data_.data().get()->async_gpu_push(stream);  
        // ����Ƿ�ɹ�  
        CUDA_CHECK(cudaStreamSynchronize(stream));  
      }  
#endif  
      // ��װ�õ�batchѹ��full����  
      prefetch_full_.push(batch);  
    }  
  } catch (boost::thread_interrupted&) {  
    // Interrupted exception is expected on shutdown  
  }  
#ifndef CPU_ONLY  
  if (Caffe::mode() == Caffe::GPU) {  
      // ������  
    CUDA_CHECK(cudaStreamDestroy(stream));  
  }  
#endif  
}  
  
template <typename Dtype>  
void BasePrefetchingDataLayer<Dtype>::Forward_cpu(  
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {  
    // ���ݵ�ʱ���Ǵ�full�����е���һ������  
  Batch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");  
  // Reshape to loaded data.  
  // ����batch����״�ı�������״  
  top[0]->ReshapeLike(batch->data_);  
  // Copy the data  
  // ��batch���ݸ��Ƶ�top[0]  
  caffe_copy(batch->data_.count(), batch->data_.cpu_data(),  
             top[0]->mutable_cpu_data());  
  DLOG(INFO) << "Prefetch copied";  
  if (this->output_labels_) {  
      // ������Ļ�  
    // Reshape to loaded labels.  
    // ����batch��������״�ı�top[1]����״  
    top[1]->ReshapeLike(batch->label_);  
    // Copy the labels.  
    // ������굽top[1]  
    caffe_copy(batch->label_.count(), batch->label_.cpu_data(),  
        top[1]->mutable_cpu_data());  
  }  
  // ����batchѹ��free����  
  prefetch_free_.push(batch);  
}  
  
  
// ���û��GPU�Ļ�����BasePrefetchingDataLayer��������һ��Forward����  
// �ú�������ǰ��������ֱ�ӱ���  
#ifdef CPU_ONLY  
STUB_GPU_FORWARD(BasePrefetchingDataLayer, Forward);  
#endif  
// ��ʼ����  
INSTANTIATE_CLASS(BaseDataLayer);
INSTANTIATE_CLASS(BasePrefetchingDataLayer);

}  // namespace caffe
