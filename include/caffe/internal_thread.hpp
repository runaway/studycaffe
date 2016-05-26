#ifndef CAFFE_INTERNAL_THREAD_HPP_
#define CAFFE_INTERNAL_THREAD_HPP_

#include "caffe/common.hpp"
// internal_thread.hpp�����װ��pthread�������̳е�������Եõ�һ���������̣߳���Ҫ�������ڼ��㵱ǰ��һ������ʱ���ں�̨��ȡ��һ�������ݡ�
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
// InternalThread��ʵ���Ͼ���boost���thread�ķ�װ
class InternalThread {
 public:
  InternalThread() : thread_() {}
  virtual ~InternalThread();

  /**
   * Caffe's thread local state will be initialized using the current
   * thread values, e.g. device id, solver index etc. The random seed
   * is initialized using caffe_rng_rand.
   */
  // caffe���ֲ߳̾�״̬����ʹ�õ�ǰ�߳�ֵ�����г�ʼ������ǰ���̵߳�ֵ���豸id��solver�ı�š���������ӵ� 
  void StartInternalThread();

  /** Will not return until the internal thread has exited. */
  // �Ƿ�֪���߳��˳��ŷ��� 
  void StopInternalThread();

  // �߳��Ƿ��Ѿ�������
  bool is_started() const;

 protected:
  /* Implement this method in your subclass
      with the code you want your thread to run. */
  // ������һ���麯����Ҫ��̳и���ı���Ҫʵ��֮  
  virtual void InternalThreadEntry() {}

  /* Should be tested when running loops to exit when requested. */
  // �ڵ������˳���ʱ��Ӧ�õ��øú��� 
  bool must_stop();

 private:
  void entry(int device, Caffe::Brew mode, int rand_seed, int solver_count,
      bool root_solver);

  shared_ptr<boost::thread> thread_;
};

}  // namespace caffe

#endif  // CAFFE_INTERNAL_THREAD_HPP_
