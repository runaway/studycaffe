#include <boost/thread.hpp>
#include <exception>

#include "caffe/internal_thread.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

InternalThread::~InternalThread() 
{
    StopInternalThread();
}

bool InternalThread::is_started() const 
{
    // 首先thread_指针不能为空，然后该线程是可等待的（joinable）  
    return thread_ && thread_->joinable();
}

bool InternalThread::must_stop() {
  return thread_ && thread_->interruption_requested();
}

// 初始化工作，然后 
void InternalThread::StartInternalThread()
{
    CHECK(!is_started()) << "Threads should persist and not be restarted.";

    int device = 0;
    
#ifndef CPU_ONLY
    CUDA_CHECK(cudaGetDevice(&device));
#endif

    Caffe::Brew mode = Caffe::mode();
    int rand_seed = caffe_rng_rand();
    int solver_count = Caffe::solver_count();
    bool root_solver = Caffe::root_solver();

    try 
    {
        // 然后重新实例化一个thread对象给thread_指针，该线程的执行的是entry函数  
        thread_.reset(new boost::thread(&InternalThread::entry, this, device, mode,
              rand_seed, solver_count, root_solver));
    } 
    catch (std::exception& e) 
    {
        LOG(FATAL) << "Thread exception: " << e.what();
    }
}

// 线程所要执行的函数
void InternalThread::entry(int device, Caffe::Brew mode, int rand_seed,
    int solver_count, bool root_solver) 
{

#ifndef CPU_ONLY
    CUDA_CHECK(cudaSetDevice(device));
#endif
    Caffe::set_mode(mode);
    Caffe::set_random_seed(rand_seed);
    Caffe::set_solver_count(solver_count);
    Caffe::set_root_solver(root_solver);
    
    /*
    关于两个StartInternalThread函数, 调用了InternalThreadEntry()函数.而这个函数里有 
    while (!must_stop()){xxxfree_.pop()}. 这个while循环会一直进行下去,直到调用析构函数, 从而StopInternalThread(). 
    */
    InternalThreadEntry();
}

// 停止线程
void InternalThread::StopInternalThread() 
{
    // 如果线程已经开始  
    if (is_started()) 
    {
        // 那么打断  
        thread_->interrupt();
        
        try 
        {
            // 等待线程结束  
            thread_->join();
        } 
        catch (boost::thread_interrupted&) 
        { 
            //如果被打断，啥也不干，因为是自己要打断的^_^  
        } 
        catch (std::exception& e) 
        { 
            // 如果发生其他错误则记录到日志  
            LOG(FATAL) << "Thread exception: " << e.what();
        }
    }
}

// 无非就是获取线程的状态、启动线程、以及定义的线程入口函数InternalThread::entry，
// 这个入口函数很有意思，里面调用了虚函数InternalThreadEntry，并且在调用之前,帮
// 用户做好了初始化的工作（随机数种子，CUDA、工作模式及GPU还是CPU、solver的类型）。

}  // namespace caffe
