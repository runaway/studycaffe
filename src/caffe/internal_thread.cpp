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
    // ����thread_ָ�벻��Ϊ�գ�Ȼ����߳��ǿɵȴ��ģ�joinable��  
    return thread_ && thread_->joinable();
}

bool InternalThread::must_stop() {
  return thread_ && thread_->interruption_requested();
}

// ��ʼ��������Ȼ�� 
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
        // Ȼ������ʵ����һ��thread�����thread_ָ�룬���̵߳�ִ�е���entry����  
        thread_.reset(new boost::thread(&InternalThread::entry, this, device, mode,
              rand_seed, solver_count, root_solver));
    } 
    catch (std::exception& e) 
    {
        LOG(FATAL) << "Thread exception: " << e.what();
    }
}

// �߳���Ҫִ�еĺ���
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
    ��������StartInternalThread����, ������InternalThreadEntry()����.������������� 
    while (!must_stop()){xxxfree_.pop()}. ���whileѭ����һֱ������ȥ,ֱ��������������, �Ӷ�StopInternalThread(). 
    */
    InternalThreadEntry();
}

// ֹͣ�߳�
void InternalThread::StopInternalThread() 
{
    // ����߳��Ѿ���ʼ  
    if (is_started()) 
    {
        // ��ô���  
        thread_->interrupt();
        
        try 
        {
            // �ȴ��߳̽���  
            thread_->join();
        } 
        catch (boost::thread_interrupted&) 
        { 
            //�������ϣ�ɶҲ���ɣ���Ϊ���Լ�Ҫ��ϵ�^_^  
        } 
        catch (std::exception& e) 
        { 
            // ������������������¼����־  
            LOG(FATAL) << "Thread exception: " << e.what();
        }
    }
}

// �޷Ǿ��ǻ�ȡ�̵߳�״̬�������̡߳��Լ�������߳���ں���InternalThread::entry��
// �����ں���������˼������������麯��InternalThreadEntry�������ڵ���֮ǰ,��
// �û������˳�ʼ���Ĺ�������������ӣ�CUDA������ģʽ��GPU����CPU��solver�����ͣ���

}  // namespace caffe
