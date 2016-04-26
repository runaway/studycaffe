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
首先介绍一下boost::weak_ptr;
弱引用是为了解决shared_ptr在循环引用下的内存释放问题而产生的。
弱引用当引用的对象活着的时候不一定存在。仅仅是当它存在的时候的一个引用。弱引用并不修改该对象的引用计数，这意味这弱引用它并不对对象的内存进行管理，在功能上类似于普通指针，然而一个比较大的区别是，弱引用能检测到所管理的对象是否已经被释放，从而避免访问非法内存。
由于弱引用不更改引用计数，类似普通指针，只要把循环引用的一方使用弱引用，即可解除循环引用。
*/
using boost::weak_ptr;

map<const string, weak_ptr<DataReader::Body> > DataReader::bodies_;
static boost::mutex bodies_mutex_;

// 构造函数，传入的是网络的参数、  
// 初始化queue_pair_（里面包含两个阻塞队列free_和full_）  
DataReader::DataReader(const LayerParameter& param)  
    : queue_pair_(new QueuePair(  //  
        param.data_param().prefetch() * param.data_param().batch_size())) {  
  // Get or create a body  
  // 首先创建或者获取一个body实例  
  boost::mutex::scoped_lock lock(bodies_mutex_);  
  string key = source_key(param);// 从网络参数中获取key  
  weak_ptr<Body>& weak = bodies_[key];// bodies_是存放的string到Body的映射  
  body_ = weak.lock();  
  if (!body_) {// 如果bodies是空的  
    body_.reset(new Body(param));// 则新建Body实例到body_  
    bodies_[key] = weak_ptr<Body>(body_);// 然后存放到bodies_中去  
  }  
  body_->new_queue_pairs_.push(queue_pair_); // 并将queue_pair放入body_中的new_queue_pairs_中去  
}  
// 析构函数  
DataReader::~DataReader() {  
  string key = source_key(body_->param_);  
  body_.reset();  
  boost::mutex::scoped_lock lock(bodies_mutex_);// 上锁  
  if (bodies_[key].expired()) {  
    bodies_.erase(key);// map里面的erase  
  }  
}  

//
// 内部还定义了一个QueuePair类，该类有free和full函数，该类用于在body和readers之间进行数据分享
// 根据给定的size初始化的若干个Datum（本文最后会给出该数据结构的定义）的实例到free里面
DataReader::QueuePair::QueuePair(int size) {
  // Initialize the free queue with requested number of datums
  // 一开始全部压入free  
  for (int i = 0; i < size; ++i) {
    free_.push(new Datum());
  }
}

// 将full_和free_这两个队列里面的Datum对象全部delete
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
// 内部还定义了一个Body类，该类是继承于InternalThread
// Body里面重写了InternalThread内部的InternalThreadEntry函数，此外还添加了read_one函数
// Body内部有DataReader的友元，以及BlockingQueue<shared_ptr<QueuePair> > new_queue_pairs_;

DataReader::Body::Body(const LayerParameter& param)
    : param_(param),
      new_queue_pairs_() {
    // Body类的构造函数，实际上是给定网络的参数，然后开始启动内部线程
  StartInternalThread();
}

DataReader::Body::~Body() {
    // 析构，停止线程  
  StopInternalThread();
}

// 自己实现的需要执行的函数  
// 首先打开数据库，然后设置游标，然后设置QueuePair指针容器  
void DataReader::Body::InternalThreadEntry() {
    // 获取所给定的数据源的类型来得到DB的指针
  shared_ptr<db::DB> db(db::GetDB(param_.data_param().backend()));

  // 从网络参数中给定的DB的位置打开DB  
  db->Open(param_.data_param().source(), db::READ);

  // 新建游标指针
  shared_ptr<db::Cursor> cursor(db->NewCursor());

  // 新建QueuePair指针容器，QueuePair里面包含了free_和full_这两个阻塞队列  
  vector<shared_ptr<QueuePair> > qps;
  try {
    // 根据网络参数的阶段来设置solver_count  
    int solver_count = param_.phase() == TRAIN ? Caffe::solver_count() : 1;

    // To ensure deterministic runs, only start running once all solvers
    // are ready. But solvers need to peek on one item during initialization,
    // so read one item, then wait for the next solver.
    for (int i = 0; i < solver_count; ++i) {
      shared_ptr<QueuePair> qp(new_queue_pairs_.pop());
      read_one(cursor.get(), qp.get()); // 读取一个数据  
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

// 从数据库中获取一个数据
void DataReader::Body::read_one(db::Cursor* cursor, QueuePair* qp) {
    // 从QueuePair中的free_队列pop出一个  
  Datum* datum = qp->free_.pop();
  // TODO deserialize in-place instead of copy?
   // 然后解析cursor中的值  
  datum->ParseFromString(cursor->value());

  // 然后压入QueuePair中的full_队列  
  qp->full_.push(datum);

  // go to the next iter
  // 游标指向下一个  
  cursor->Next();
  if (!cursor->valid()) {
    DLOG(INFO) << "Restarting data prefetching from start.";
    // 如果游标指向的位置已经无效了则指向第一个位置  
    cursor->SeekToFirst();
  }
}

/*
总结：实际上该数据层就是调用了封装层的DB来读取数据，此外还简单封装了boost的线程库，然后自己封装了个阻塞队列。

最后还有Datum究竟是哈
可以看caffe.proto文件中的定义

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
