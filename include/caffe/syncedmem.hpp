#ifndef CAFFE_SYNCEDMEM_HPP_
#define CAFFE_SYNCEDMEM_HPP_

#include <cstdlib>

#include "caffe/common.hpp"

namespace caffe {

// If CUDA is available and in GPU mode, host memory will be allocated pinned,
// using cudaMallocHost. It avoids dynamic pinning for transfers (DMA).
// The improvement in performance seems negligible in the single GPU case,
// but might be more significant for parallel training. Most importantly,
// it improved stability for large models on many GPUs.
inline void CaffeMallocHost(void** ptr, size_t size, bool* use_cuda) {
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaMallocHost(ptr, size));
    *use_cuda = true;
    return;
  }
#endif
  *ptr = malloc(size);
  *use_cuda = false;
  CHECK(*ptr) << "host allocation of size " << size << " failed";
}

inline void CaffeFreeHost(void* ptr, bool use_cuda) {
#ifndef CPU_ONLY
  if (use_cuda) {
    CUDA_CHECK(cudaFreeHost(ptr));
    return;
  }
#endif
  free(ptr);
}

/*
Caffe中的Blob是其进行数据传递与处理的一个类，经过分析Blob中的数据（data_、diff_、
shape_data_）都是SyncedMemory的类型，本文简单的分析了SyncedMemory和Blob这两个类
的Public函数。
*/
/**
 * @brief Manages memory allocation and synchronization between the host (CPU)
 *        and device (GPU).
 *
 * TODO(dox): more thorough description.
 */
// 管理内存数据以及CPU和GPU之间的内存同步 
class SyncedMemory 
{
public:
    SyncedMemory()
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), head_(UNINITIALIZED),
        own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false),
        gpu_device_(-1) {}
    explicit SyncedMemory(size_t size)
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
        own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false),
        gpu_device_(-1) {}
    ~SyncedMemory();

    // 获取CPUDATA  
    // 获取cpu数据，在该函数的实现中，会首先调用to_cpu()，将数据转换成cpu数据，
    // 然后返回cpu数据的指针。
    const void* cpu_data();  

    // 设置cpu数据
    void set_cpu_data(void* data);  

    // 获取GPU数据，在该函数的实现中，会首先调用to_gpu()，将数据转换成gpu数据，
    // 然后返回gpu数据的指针。
    const void* gpu_data();  

    // 设置GPU数据
    void set_gpu_data(void* data);  

    // 获取互斥的CPU或者GPUDATA  

    // 同cpu_data()，只不过该函数返回的数据是可写的
    void* mutable_cpu_data(); 

    // 同gpu_data()，只不过该函数返回的数据是可写的
    void* mutable_gpu_data();  

    // 枚举类型，未初始化，在CPU、在GPU、同步状态  
    // 用于表示数据在CPU和GPU之间传输的状态
    enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };  
    
    // 获取数据的位置  
    // 获取同步头，该头用于表示数据在CPU和GPU之间传输的状态
    SyncedHead head() 
    { 
        return head_; 
    }
    
    // 获取数据buffer的大小
    size_t size() 
    { 
        return size_; 
    }  

    // 向GPU异步push数据
#ifndef CPU_ONLY  
    void async_gpu_push(const cudaStream_t& stream);  
#endif  

private:  
    // 内部使用的到cpu还是gpu  
    void to_cpu();  
    void to_gpu();  
    void* cpu_ptr_;
    void* gpu_ptr_;
    size_t size_;
    SyncedHead head_;
    bool own_cpu_data_;
    bool cpu_malloc_use_cuda_;
    bool own_gpu_data_;
    int gpu_device_;

    DISABLE_COPY_AND_ASSIGN(SyncedMemory);
};  // class SyncedMemory

}  // namespace caffe

#endif  // CAFFE_SYNCEDMEM_HPP_
