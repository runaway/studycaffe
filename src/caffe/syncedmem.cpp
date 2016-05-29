#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
// 析构函数就是释放内存  
SyncedMemory::~SyncedMemory() {  
  if (cpu_ptr_ && own_cpu_data_) {  
    CaffeFreeHost(cpu_ptr_, 0);  
  }  
  
#ifndef CPU_ONLY// 只要不是定义的CPU_ONLY的编译模式  
  if (gpu_ptr_ && own_gpu_data_) {  
    int initial_device;  
    // 获取可用设备  
    cudaGetDevice(&initial_device);  
    if (gpu_device_ != -1) {  
        // 当前所使用的设备  
      CUDA_CHECK(cudaSetDevice(gpu_device_));  
    }  
    // 释放当前  
    CUDA_CHECK(cudaFree(gpu_ptr_));  
    cudaSetDevice(initial_device);  
  }  
#endif  // CPU_ONLY  
}  

/*
功能：把数据放到cpu上 
1.数据未初始化，则在cpu申请内存（申请为0）。此时状态为HEAD_AT_CPU 
2.数据本来在gpu，则从gpu拷贝内存到cpu。此时状态为SYNCED 
3.数据本来在cpu，不做处理 
4.数据在cpu和gpu都有，不做处理
*/
// 内部使用的  
// 如果当前未初始化，直接在内存分配空间  
// 如果在GPU上则复制到内存  
// 如果已经在内存则啥都不动  
inline void SyncedMemory::to_cpu() {  
  switch (head_) {  
  // 如果当前是未初始化，直接分配CPU上的内存  
  case UNINITIALIZED:
    CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
    caffe_memset(size_, 0, cpu_ptr_);
    head_ = HEAD_AT_CPU;
    own_cpu_data_ = true;
    break;
  case HEAD_AT_GPU:
#ifndef CPU_ONLY
    // 如果当前数据在GPU，然后cpu_ptr为空  
    if (cpu_ptr_ == NULL) {  
        // 分配内存  
      CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
      own_cpu_data_ = true;
    }
    // 复制数据  
    caffe_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_);  
    head_ = SYNCED;  
#else// CPU_ONLY模式当然只能报错了  
    NO_GPU;  
#endif
    break;
  case HEAD_AT_CPU:
  case SYNCED:
    break;
  }
}


/*
功能：把数据放到gpu上 
1.数据未初始化，在gpu申请内存（申请为0）。此时状态为HEAD_AT_GPU 
2.数据在cpu，从cpu拷贝到gpu。此时状态为SYNCED 
3.数据在gpu，不做操作。 
4.数据在cpu和gpu都有，不做操作。
*/
// 内部使用的  
// 如果当前未初始化直接在GPU分配内存  
// 如果当前在CPU，则在GPU上分配内存并且复制到GPU  
// 如果数据已经在GPU则啥也不做  
inline void SyncedMemory::to_gpu() {  
#ifndef CPU_ONLY  
  switch (head_) {  
  case UNINITIALIZED:  
    // 获取设备  
    CUDA_CHECK(cudaGetDevice(&gpu_device_));  
    // 在设备上分配内存  
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));  
    // 初始化为0  
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

// 功能：返回数据在cpu的指针  
// 首先不管三七二十一将数据搞到内存上去  
// 然后获取cpu上的数据  
const void* SyncedMemory::cpu_data() {  
  to_cpu();  
  return (const void*)cpu_ptr_;  
}  
  
  
// 如果当前cpu_ptr_有内存上的数据则先释放  
// 然后再将地址给内部变量cpu_ptr_  
// 设置cpu上的数据  
void SyncedMemory::set_cpu_data(void* data) {
  CHECK(data);
  if (own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  }
  cpu_ptr_ = data;
  head_ = HEAD_AT_CPU;
  own_cpu_data_ = false;
}

// 首先不管三七二十一将数据搞到GPU上去  
// 然后在获取gpu上的数据  
// 但是并没有改变head_的值(head_表明数据究竟在哪儿)  
const void* SyncedMemory::gpu_data() {  
#ifndef CPU_ONLY  
  to_gpu();
  return (const void*)gpu_ptr_;
#else
  NO_GPU;
  return NULL;
#endif
}

// 如果当前gpu_ptr_有内存上的数据则先释放  
// 然后再将地址给内部变量gpu_ptr_  
// 设置gpu上的数据  
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
// 功能：返回数据在cpu的指针，并改变数据的状态为HEAD_AT_CPU
// 首先不管三七二十一先数据搞到CPU上去  
// 然后返回互斥的cpu_ptr_指针  
// mutable_cpu_data与cpu_data的区别就是是否设置head  
void* SyncedMemory::mutable_cpu_data() {  
  to_cpu();  
  head_ = HEAD_AT_CPU;  
  return cpu_ptr_;  
}  
  
// 功能：返回数据在cpu的指针，并改变数据的状态为HEAD_AT_GPU  
// 首先不管三七二十一先数据搞到GPU上去  
// 然后返回互斥的gpu_ptr_指针  
// mutable_gpu_data与gpu_data的区别就是是否设置head  
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
// 异步推送数据到gpu  
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

