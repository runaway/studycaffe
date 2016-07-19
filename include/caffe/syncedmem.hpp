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
Caffe�е�Blob����������ݴ����봦���һ���࣬��������Blob�е����ݣ�data_��diff_��
shape_data_������SyncedMemory�����ͣ����ļ򵥵ķ�����SyncedMemory��Blob��������
��Public������
*/
/**
 * @brief Manages memory allocation and synchronization between the host (CPU)
 *        and device (GPU).
 *
 * TODO(dox): more thorough description.
 */
// �����ڴ������Լ�CPU��GPU֮����ڴ�ͬ�� 
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

    // ��ȡCPUDATA  
    // ��ȡcpu���ݣ��ڸú�����ʵ���У������ȵ���to_cpu()��������ת����cpu���ݣ�
    // Ȼ�󷵻�cpu���ݵ�ָ�롣
    const void* cpu_data();  

    // ����cpu����
    void set_cpu_data(void* data);  

    // ��ȡGPU���ݣ��ڸú�����ʵ���У������ȵ���to_gpu()��������ת����gpu���ݣ�
    // Ȼ�󷵻�gpu���ݵ�ָ�롣
    const void* gpu_data();  

    // ����GPU����
    void set_gpu_data(void* data);  

    // ��ȡ�����CPU����GPUDATA  

    // ͬcpu_data()��ֻ�����ú������ص������ǿ�д��
    void* mutable_cpu_data(); 

    // ͬgpu_data()��ֻ�����ú������ص������ǿ�д��
    void* mutable_gpu_data();  

    // ö�����ͣ�δ��ʼ������CPU����GPU��ͬ��״̬  
    // ���ڱ�ʾ������CPU��GPU֮�䴫���״̬
    enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };  
    
    // ��ȡ���ݵ�λ��  
    // ��ȡͬ��ͷ����ͷ���ڱ�ʾ������CPU��GPU֮�䴫���״̬
    SyncedHead head() 
    { 
        return head_; 
    }
    
    // ��ȡ����buffer�Ĵ�С
    size_t size() 
    { 
        return size_; 
    }  

    // ��GPU�첽push����
#ifndef CPU_ONLY  
    void async_gpu_push(const cudaStream_t& stream);  
#endif  

private:  
    // �ڲ�ʹ�õĵ�cpu����gpu  
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
