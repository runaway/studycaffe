#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
// �������������ͷ��ڴ�  
SyncedMemory::~SyncedMemory() {  
  if (cpu_ptr_ && own_cpu_data_) {  
    CaffeFreeHost(cpu_ptr_, 0);  
  }  
  
#ifndef CPU_ONLY// ֻҪ���Ƕ����CPU_ONLY�ı���ģʽ  
  if (gpu_ptr_ && own_gpu_data_) {  
    int initial_device;  
    // ��ȡ�����豸  
    cudaGetDevice(&initial_device);  
    if (gpu_device_ != -1) {  
        // ��ǰ��ʹ�õ��豸  
      CUDA_CHECK(cudaSetDevice(gpu_device_));  
    }  
    // �ͷŵ�ǰ  
    CUDA_CHECK(cudaFree(gpu_ptr_));  
    cudaSetDevice(initial_device);  
  }  
#endif  // CPU_ONLY  
}  

/*
���ܣ������ݷŵ�cpu�� 
1.����δ��ʼ��������cpu�����ڴ棨����Ϊ0������ʱ״̬ΪHEAD_AT_CPU 
2.���ݱ�����gpu�����gpu�����ڴ浽cpu����ʱ״̬ΪSYNCED 
3.���ݱ�����cpu���������� 
4.������cpu��gpu���У���������
*/
// �ڲ�ʹ�õ�  
// �����ǰδ��ʼ����ֱ�����ڴ����ռ�  
// �����GPU�����Ƶ��ڴ�  
// ����Ѿ����ڴ���ɶ������  
inline void SyncedMemory::to_cpu() {  
  switch (head_) {  
  // �����ǰ��δ��ʼ����ֱ�ӷ���CPU�ϵ��ڴ�  
  case UNINITIALIZED:
    CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
    caffe_memset(size_, 0, cpu_ptr_);
    head_ = HEAD_AT_CPU;
    own_cpu_data_ = true;
    break;
  case HEAD_AT_GPU:
#ifndef CPU_ONLY
    // �����ǰ������GPU��Ȼ��cpu_ptrΪ��  
    if (cpu_ptr_ == NULL) {  
        // �����ڴ�  
      CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
      own_cpu_data_ = true;
    }
    // ��������  
    caffe_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_);  
    head_ = SYNCED;  
#else// CPU_ONLYģʽ��Ȼֻ�ܱ�����  
    NO_GPU;  
#endif
    break;
  case HEAD_AT_CPU:
  case SYNCED:
    break;
  }
}


/*
���ܣ������ݷŵ�gpu�� 
1.����δ��ʼ������gpu�����ڴ棨����Ϊ0������ʱ״̬ΪHEAD_AT_GPU 
2.������cpu����cpu������gpu����ʱ״̬ΪSYNCED 
3.������gpu������������ 
4.������cpu��gpu���У�����������
*/
// �ڲ�ʹ�õ�  
// �����ǰδ��ʼ��ֱ����GPU�����ڴ�  
// �����ǰ��CPU������GPU�Ϸ����ڴ沢�Ҹ��Ƶ�GPU  
// ��������Ѿ���GPU��ɶҲ����  
inline void SyncedMemory::to_gpu() {  
#ifndef CPU_ONLY  
  switch (head_) {  
  case UNINITIALIZED:  
    // ��ȡ�豸  
    CUDA_CHECK(cudaGetDevice(&gpu_device_));  
    // ���豸�Ϸ����ڴ�  
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));  
    // ��ʼ��Ϊ0  
    caffe_gpu_memset(size_, 0, gpu_ptr_);  
    head_ = HEAD_AT_GPU;  
    own_gpu_data_ = true;  
    break;  
  case HEAD_AT_CPU:  
    if (gpu_ptr_ == NULL) {  
      CUDA_CHECK(cudaGetDevice(&gpu_device_));  
      CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));  
      own_gpu_data_ = true;  
    }  
    caffe_gpu_memcpy(size_, cpu_ptr_, gpu_ptr_);  
    head_ = SYNCED;  
    break;  
  case HEAD_AT_GPU:  
  case SYNCED:  
    break;  
  }  
#else  
  NO_GPU;  
#endif  
}  

// ���ܣ�����������cpu��ָ��  
// ���Ȳ������߶�ʮһ�����ݸ㵽�ڴ���ȥ  
// Ȼ���ȡcpu�ϵ�����  
const void* SyncedMemory::cpu_data() {  
  to_cpu();  
  return (const void*)cpu_ptr_;  
}  
  
  
// �����ǰcpu_ptr_���ڴ��ϵ����������ͷ�  
// Ȼ���ٽ���ַ���ڲ�����cpu_ptr_  
// ����cpu�ϵ�����  
void SyncedMemory::set_cpu_data(void* data) {
  CHECK(data);
  if (own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  }
  cpu_ptr_ = data;
  head_ = HEAD_AT_CPU;
  own_cpu_data_ = false;
}

// ���Ȳ������߶�ʮһ�����ݸ㵽GPU��ȥ  
// Ȼ���ڻ�ȡgpu�ϵ�����  
// ���ǲ�û�иı�head_��ֵ(head_�������ݾ������Ķ�)  
const void* SyncedMemory::gpu_data() {  
#ifndef CPU_ONLY  
  to_gpu();
  return (const void*)gpu_ptr_;
#else
  NO_GPU;
  return NULL;
#endif
}

// �����ǰgpu_ptr_���ڴ��ϵ����������ͷ�  
// Ȼ���ٽ���ַ���ڲ�����gpu_ptr_  
// ����gpu�ϵ�����  
void SyncedMemory::set_gpu_data(void* data) {  
#ifndef CPU_ONLY  
  CHECK(data);
  if (own_gpu_data_) {
    int initial_device;
    cudaGetDevice(&initial_device);
    if (gpu_device_ != -1) {
      CUDA_CHECK(cudaSetDevice(gpu_device_));
    }
    CUDA_CHECK(cudaFree(gpu_ptr_));
    cudaSetDevice(initial_device);
  }
  gpu_ptr_ = data;
  head_ = HEAD_AT_GPU;
  own_gpu_data_ = false;
#else
  NO_GPU;
#endif
}
// ���ܣ�����������cpu��ָ�룬���ı����ݵ�״̬ΪHEAD_AT_CPU
// ���Ȳ������߶�ʮһ�����ݸ㵽CPU��ȥ  
// Ȼ�󷵻ػ����cpu_ptr_ָ��  
// mutable_cpu_data��cpu_data����������Ƿ�����head  
void* SyncedMemory::mutable_cpu_data() {  
  to_cpu();  
  head_ = HEAD_AT_CPU;  
  return cpu_ptr_;  
}  
  
// ���ܣ�����������cpu��ָ�룬���ı����ݵ�״̬ΪHEAD_AT_GPU  
// ���Ȳ������߶�ʮһ�����ݸ㵽GPU��ȥ  
// Ȼ�󷵻ػ����gpu_ptr_ָ��  
// mutable_gpu_data��gpu_data����������Ƿ�����head  
void* SyncedMemory::mutable_gpu_data() {  
#ifndef CPU_ONLY  
  to_gpu();  
  head_ = HEAD_AT_GPU;  
  return gpu_ptr_;  
#else  
  NO_GPU;  
#endif  
}  
  
#ifndef CPU_ONLY  
// �첽�������ݵ�gpu  
void SyncedMemory::async_gpu_push(const cudaStream_t& stream) {  
  CHECK(head_ == HEAD_AT_CPU);  
  if (gpu_ptr_ == NULL) {  
    CUDA_CHECK(cudaGetDevice(&gpu_device_));  
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));  
    own_gpu_data_ = true;  
  }
  const cudaMemcpyKind put = cudaMemcpyHostToDevice;
  CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_, put, stream));
  // Assume caller will synchronize on the stream before use
  head_ = SYNCED;
}
#endif

}  // namespace caffe

