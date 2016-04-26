#include <boost/thread.hpp>
#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
/*
���Ƚ���һ��boost::weak_ptr;
��������Ϊ�˽��shared_ptr��ѭ�������µ��ڴ��ͷ�����������ġ�
�����õ����õĶ�����ŵ�ʱ��һ�����ڡ������ǵ������ڵ�ʱ���һ�����á������ò����޸ĸö�������ü���������ζ���������������Զ�����ڴ���й����ڹ�������������ָͨ�룬Ȼ��һ���Ƚϴ�������ǣ��������ܼ�⵽������Ķ����Ƿ��Ѿ����ͷţ��Ӷ�������ʷǷ��ڴ档
���������ò��������ü�����������ָͨ�룬ֻҪ��ѭ�����õ�һ��ʹ�������ã����ɽ��ѭ�����á�
*/
using boost::weak_ptr;

map<const string, weak_ptr<DataReader::Body> > DataReader::bodies_;
static boost::mutex bodies_mutex_;

// ���캯���������������Ĳ�����  
// ��ʼ��queue_pair_���������������������free_��full_��  
DataReader::DataReader(const LayerParameter& param)  
    : queue_pair_(new QueuePair(  //  
        param.data_param().prefetch() * param.data_param().batch_size())) {  
  // Get or create a body  
  // ���ȴ������߻�ȡһ��bodyʵ��  
  boost::mutex::scoped_lock lock(bodies_mutex_);  
  string key = source_key(param);// ����������л�ȡkey  
  weak_ptr<Body>& weak = bodies_[key];// bodies_�Ǵ�ŵ�string��Body��ӳ��  
  body_ = weak.lock();  
  if (!body_) {// ���bodies�ǿյ�  
    body_.reset(new Body(param));// ���½�Bodyʵ����body_  
    bodies_[key] = weak_ptr<Body>(body_);// Ȼ���ŵ�bodies_��ȥ  
  }  
  body_->new_queue_pairs_.push(queue_pair_); // ����queue_pair����body_�е�new_queue_pairs_��ȥ  
}  
// ��������  
DataReader::~DataReader() {  
  string key = source_key(body_->param_);  
  body_.reset();  
  boost::mutex::scoped_lock lock(bodies_mutex_);// ����  
  if (bodies_[key].expired()) {  
    bodies_.erase(key);// map�����erase  
  }  
}  

//
// �ڲ���������һ��QueuePair�࣬������free��full����������������body��readers֮��������ݷ���
// ���ݸ�����size��ʼ�������ɸ�Datum������������������ݽṹ�Ķ��壩��ʵ����free����
DataReader::QueuePair::QueuePair(int size) {
  // Initialize the free queue with requested number of datums
  // һ��ʼȫ��ѹ��free  
  for (int i = 0; i < size; ++i) {
    free_.push(new Datum());
  }
}

// ��full_��free_���������������Datum����ȫ��delete
DataReader::QueuePair::~QueuePair() {
  Datum* datum;
  while (free_.try_pop(&datum)) {
    delete datum;
  }
  while (full_.try_pop(&datum)) {
    delete datum;
  }
}

//
// �ڲ���������һ��Body�࣬�����Ǽ̳���InternalThread
// Body������д��InternalThread�ڲ���InternalThreadEntry���������⻹�����read_one����
// Body�ڲ���DataReader����Ԫ���Լ�BlockingQueue<shared_ptr<QueuePair> > new_queue_pairs_;

DataReader::Body::Body(const LayerParameter& param)
    : param_(param),
      new_queue_pairs_() {
    // Body��Ĺ��캯����ʵ�����Ǹ�������Ĳ�����Ȼ��ʼ�����ڲ��߳�
  StartInternalThread();
}

DataReader::Body::~Body() {
    // ������ֹͣ�߳�  
  StopInternalThread();
}

// �Լ�ʵ�ֵ���Ҫִ�еĺ���  
// ���ȴ����ݿ⣬Ȼ�������α꣬Ȼ������QueuePairָ������  
void DataReader::Body::InternalThreadEntry() {
    // ��ȡ������������Դ���������õ�DB��ָ��
  shared_ptr<db::DB> db(db::GetDB(param_.data_param().backend()));

  // ����������и�����DB��λ�ô�DB  
  db->Open(param_.data_param().source(), db::READ);

  // �½��α�ָ��
  shared_ptr<db::Cursor> cursor(db->NewCursor());

  // �½�QueuePairָ��������QueuePair���������free_��full_��������������  
  vector<shared_ptr<QueuePair> > qps;
  try {
    // ������������Ľ׶�������solver_count  
    int solver_count = param_.phase() == TRAIN ? Caffe::solver_count() : 1;

    // To ensure deterministic runs, only start running once all solvers
    // are ready. But solvers need to peek on one item during initialization,
    // so read one item, then wait for the next solver.
    for (int i = 0; i < solver_count; ++i) {
      shared_ptr<QueuePair> qp(new_queue_pairs_.pop());
      read_one(cursor.get(), qp.get()); // ��ȡһ������  
      qps.push_back(qp);
    }
    // Main loop
    while (!must_stop()) {
      for (int i = 0; i < solver_count; ++i) {
        read_one(cursor.get(), qps[i].get());
      }
      // Check no additional readers have been created. This can happen if
      // more than one net is trained at a time per process, whether single
      // or multi solver. It might also happen if two data layers have same
      // name and same source.
      CHECK_EQ(new_queue_pairs_.size(), 0);
    }
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
}

// �����ݿ��л�ȡһ������
void DataReader::Body::read_one(db::Cursor* cursor, QueuePair* qp) {
    // ��QueuePair�е�free_����pop��һ��  
  Datum* datum = qp->free_.pop();
  // TODO deserialize in-place instead of copy?
   // Ȼ�����cursor�е�ֵ  
  datum->ParseFromString(cursor->value());

  // Ȼ��ѹ��QueuePair�е�full_����  
  qp->full_.push(datum);

  // go to the next iter
  // �α�ָ����һ��  
  cursor->Next();
  if (!cursor->valid()) {
    DLOG(INFO) << "Restarting data prefetching from start.";
    // ����α�ָ���λ���Ѿ���Ч����ָ���һ��λ��  
    cursor->SeekToFirst();
  }
}

/*
�ܽ᣺ʵ���ϸ����ݲ���ǵ����˷�װ���DB����ȡ���ݣ����⻹�򵥷�װ��boost���߳̿⣬Ȼ���Լ���װ�˸��������С�

�����Datum�����ǹ�
���Կ�caffe.proto�ļ��еĶ���

message Datum {
  optional int32 channels = 1;
  optional int32 height = 2;
  optional int32 width = 3;
  // the actual image data, in bytes
  optional bytes data = 4;
  optional int32 label = 5;
  // Optionally, the datum could also hold float data.
  repeated float float_data = 6;
  // If true data contains an encoded image that need to be decoded
  optional bool encoded = 7 [default = false];
}
*/

}  // namespace caffe
