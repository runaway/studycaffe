#ifndef CAFFE_INTERNAL_THREAD_HPP_
#define CAFFE_INTERNAL_THREAD_HPP_

#include "caffe/common.hpp"
// internal_thread.hpp里面封装了pthread函数，继承的子类可以得到一个单独的线程，主要作用是在计算当前的一批数据时，在后台获取新一批的数据。
/**
 Forward declare boost::thread instead of including boost/thread.hpp
 to avoid a boost/NVCC issues (#1009, #1010) on OSX.
 */
namespace boost { class thread; }

namespace caffe {

/**
 * Virtual class encapsulate boost::thread for use in base class
 * The child class will acquire the ability to run a single thread,
 * by reimplementing the virtual function InternalThreadEntry.
 */
// InternalThread类实际上就是boost库的thread的封装
class InternalThread {
 public:
  InternalThread() : thread_() {}
  virtual ~InternalThread();

  /**
   * Caffe's thread local state will be initialized using the current
   * thread values, e.g. device id, solver index etc. The random seed
   * is initialized using caffe_rng_rand.
   */
  // caffe的线程局部状态将会使用当前线程值来进行初始化，当前的线程的值有设备id，solver的编号、随机数种子等 
  void StartInternalThread();

  /** Will not return until the internal thread has exited. */
  // 是否知道线程退出才返回 
  void StopInternalThread();

  // 线程是否已经起来了
  bool is_started() const;

 protected:
  /* Implement this method in your subclass
      with the code you want your thread to run. */
  // 定义了一个虚函数，要求继承该类的必须要实现之  
  virtual void InternalThreadEntry() {}

  /* Should be tested when running loops to exit when requested. */
  // 在当请求退出的时候应该调用该函数 
  bool must_stop();

 private:
  void entry(int device, Caffe::Brew mode, int rand_seed, int solver_count,
      bool root_solver);

  shared_ptr<boost::thread> thread_;
};

}  // namespace caffe

#endif  // CAFFE_INTERNAL_THREAD_HPP_
