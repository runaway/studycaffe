#ifndef CAFFE_LAYER_H_
#define CAFFE_LAYER_H_

#include <algorithm>
#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"
/*
主要定义了一个模板类Layer
首先，看一下数据成员，主要有：
protected：
LayerParameter layer_param_ ： The protobuf that stores the layer parameters——caffe.proto文件里定义的message,相应的caffe.pb.h里定义的一个类。
Phase phase_  ：The phase: TRAIN or TEST——Phase是caffe.pb.h里定义的一个枚举类型
vector<shared_ptr<Blob<Dtype> > > blobs_ ：The vector that stores the learnable parameters as a set of blobs——所定义的向量blobs_里存储的是指向Blob<Dtyp>的智能指针，Blob<Dtype>里面存储的是learnable parameter, 使用向量是因为weight和bias是分开保存再两个blob中。
vector<bool> param_propagate_down_ ： Vector indicating whether to compute the diff of each param blob——决定是否为param blob计算梯度diff，标志每个top blob是否需要计算反向传递的梯度值。
vector<Dtype> loss_ ： The vector that indicates whether each top blob has a non-zero weight in the objective function——决定每个top blob 在 objective function是否non-zero weigh，即Losslayer中表示每个top blob计算的loss的权重。
private：
bool is_shared_ ： Whether this layer is actually shared by other nets
shared_ptr<boost::mutex> forward_mutex_ ： The mutex（互斥锁） for sequential forward if this layer is shared
然后看一下成员函数：
*/
// 其中layer.hpp是抽象出来的基类，其他都是在其基础上的继承，也即剩下的五个头文件和上图中的五个部分。

/**
 Forward declare boost::thread instead of including boost/thread.hpp
 to avoid a boost/NVCC issues (#1009, #1010) on OSX.
 */
namespace boost { class mutex; }

namespace caffe {

/**
 * @brief An interface for the units of computation which can be composed into a
 *        Net.
 *
 * Layer%s must implement a Forward function, in which they take their input
 * (bottom) Blob%s (if any) and compute their output Blob%s (if any).
 * They may also implement a Backward function, in which they compute the error
 * gradients with respect to their input Blob%s, given the error gradients with
 * their output Blob%s.
 */
template <typename Dtype>
class Layer {
 public:
    // Layer类的构建函数explicit Layer(const LayerParameter& param) : layer_param_(param)会尝试从protobuf文件读取参数。
    // 其三个主要接口：SetUp Forward Backward
    /*
SetUp函数需要根据实际的参数设置进行实现，对各种类型的参数初始化；Forward和Backward对应前向计算和反向更新，输入统一都是bottom，输出为top，其中Backward里面有个propagate_down参数，用来表示该Layer是否反向传播参数。

在Forward和Backward的具体实现里，会根据Caffe::mode()进行对应的操作，即使用cpu或者gpu进行计算，两个都实现了对应的接口Forward_cpu、Forward_gpu和Backward_cpu、Backward_gpu，这些接口都是virtual，具体还是要根据layer的类型进行对应的计算（注意：有些layer并没有GPU计算的实现，所以封装时加入了CPU的计算作为后备）。另外，还实现了ToProto的接口，将Layer的参数写入到protocol buffer文件中。
    */

// 用从protobuf 读入message LayerParameter 中的blobs 初始化 blobs_ 
// blobs_定义：vector<shared_ptr<Blob<Dtype> > > blobs_

/* 
首先获得当前网络的Phase，是train还是test，在初始化列表初始化LayerParameter,之后blobs_这里存放的是一个指向blob类的shared_ptr指针的一个vector，在这里是申请空间，然后将传入的layer_param中的blob拷贝过来。 
*/  
  // 显式的构造函数不需要重写，任何初始工作在SetUp()中完成;构造方法只是获取phase值，并且如果层说明参数(layer_param_)中提供了权值和偏置参数，也复制。    
  /**
   * You should not implement your own constructor. Any set up code should go
   * to SetUp(), where the dimensions of the bottom blobs are provided to the
   * layer.
   */
  explicit Layer(const LayerParameter& param)
    : layer_param_(param), is_shared_(false) {
      // Set phase and copy blobs (if there are any).
      // 训练还是测试？phase  
      phase_ = param.phase();

      // 在message Layerparameter中，<code>repeated BlobProto blobs</code>表示的是"The blobs containing the numeric parameters of the layer",  
      // 也就是说，在Layer中，blob存储的是参数numeric parameters，（当然参数也可以算是一种数据了，毕竟Blob是用来存储数据的）而Layer的input bottom blob以及output top blob 里面存放的才是我们通常所说的数据数据。
      if (layer_param_.blobs_size() > 0) {
        // 将blobs_的大小设置为参数中的大小  
        blobs_.resize(layer_param_.blobs_size());
        for (int i = 0; i < layer_param_.blobs_size(); ++i) {
          //blobs_的元素是指向Blob<Dtype>的智能指针,需要注意的是这句代码采用的是成员运算符，下一句代码使用的是箭头运算符。reset是因为数据类型Dtype可能会发生变化  
          // 新建若干个Blob  
          blobs_[i].reset(new Blob<Dtype>());

          //调用的是Blob类型的FromProto函数    
          // 从blob文件中获取数据  
          blobs_[i]->FromProto(layer_param_.blobs(i));
        } //读取的是权值和偏置参数  
      }
    }
  virtual ~Layer() {}

/** 
   * @brief Implements common layer setup functionality. 
   *        实现每个layer对象的setup函数 
   * 
   * @param bottom the preshaped input blobs 
   *        层的输入数据，blob中的存储空间已申请 
   * @param top 
   *     the allocated but unshaped output blobs, to be shaped by Reshape 
   *     层的输出数据，blob对象已构造但是其中的存储空间未申请，具体在Reshape函数现实现 
   * 
   * Checks that the number of bottom and top blobs is correct. 
   * Calls LayerSetUp to do special layer setup for individual layer types, 
   * followed by Reshape to set up sizes of top blobs and internal buffers. 
   * S<strong>ets up the loss weight multiplier blobs for any non-zero loss weights</strong>. 
   * This method may not be overridden. 
   * 初始化构造函数SetUp 
   * 1. 检查输入输出blob个数是否满足要求，每个层能处理的输入输出数据不一样 
   * 2. 调用LayerSetUp函数初始化特殊的层，每个Layer子类需重写这个函数完成定制的初始化 
   * 3. 调用Reshape函数为top blob分配合适大小的存储空间 
   * 4. 为每个top blob设置loss weight multiplier blobs(损失权重乘子blobs)，非LossLayer的top blob的loss weight值为零.<strong>---!!!Sets up the loss weight multiplier blobs for any non-zero loss weights!!!---</strong> 
   * 
   * 此方法非虚函数，不用重写，模式固定 
   */  

// 虚函数。会调用特定layer（子类）
// SetUp设置层的互斥量、检查BLOB的参数、调用LayerSetUp进行初始化  
// LayerSetUp是一个虚函数，用户可以去重载它。  
// 然后再设置topblob的形状以及设置损失权重。 

  void SetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
      // 初始化互斥量  
    InitMutex();
      // 检查Blob 
    CheckBlobCounts(bottom, top);
      // 层的初始化（虚函数，需用户去实现如何初始化层）  
    LayerSetUp(bottom, top);
      // 改变top的形状（虚函数，需用户去实现如何根据bottomblob改变topblob的形状）  
    Reshape(bottom, top);
       // 设置损失权重  
    SetLossWeights(top);
  }
/** 
   * @brief Does layer-specific setup: your layer should implement this function 
   *        as well as Reshape. 
   *        定制初始化，每个子类layer必须实现此虚函数！！！ 
   * 
   * @param bottom 
   *     the preshaped input blobs, whose data fields store the input data for 
   *     this layer 
   *     输入blob, 数据成员data_和diff_存储了相关数据 
   * @param top 
   *     the allocated but unshaped output blobs 
   *     输出blob, blob对象已构造但数据成员的空间尚未申请 
   * 
   * This method should do one-time layer specific setup. This includes reading 
   * and processing relevent parameters from the <code>layer_param_</code>. 
   * Setting up the shapes of top blobs and internal buffers should be done in 
   * <code>Reshape</code>, which will be called before the forward pass to 
   * adjust the top blob sizes. 
   * 此方法执行一次定制化的层初始化，包括从layer_param_读入并处理相关的层权值和偏置参数， 
   * 调用Reshape函数申请top blob的存储空间 
   */  

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  /**
   * @brief Whether a layer should be shared by multiple nets during data
   *        parallelism. By default, all layers except for data layers should
   *        not be shared. data layers should be shared to ensure each worker
   *        solver access data sequentially during data parallelism.
   */
  virtual inline bool ShareInParallel() const { return false; }

// 判断该层是否开启共享模式（即是否数据并行化了） 
  /** @brief Return whether this layer is actually shared by other nets.
   *         If ShareInParallel() is true and using more than one GPU and the
   *         net has TRAIN phase, then this function is expected return true.
   */
  inline bool IsShared() const { return is_shared_; }

 // 设置是否共享 
  /** @brief Set whether this layer is actually shared by other nets
   *         If ShareInParallel() is true and using more than one GPU and the
   *         net has TRAIN phase, then is_shared should be set true.
   */
  inline void SetShared(bool is_shared) {
    CHECK(ShareInParallel() || !is_shared)
        << type() << "Layer does not support sharing.";
    is_shared_ = is_shared;
  }
/** 
   * @brief Adjust the shapes of top blobs and internal buffers to accommodate 
   *        the shapes of the bottom blobs. 
   * 
   * @param bottom the input blobs, with the requested input shapes 
   * @param top the top blobs, which should be reshaped as needed 
   * 
   * This method should reshape top blobs as needed according to the shapes 
   * of the bottom (input) blobs, as well as reshaping any internal buffers 
   * and making any other necessary adjustments so that the layer can 
   * accommodate the bottom blobs. 
   *  
   * <strong>-----reshape top blobs 以及 internal buffers以适应bottom (input) blobs-----bottom和top都有多个blob</strong> 
   */  

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) = 0;

// 前向传播函数  
// 输入bottom，计算出top  
  /**
   * @brief Given the bottom blobs, compute the top blobs and the loss.
   *
   * @param bottom
   *     the input blobs, whose data fields store the input data for this layer
   * @param top
   *     the preshaped output blobs, whose data fields will store this layers'
   *     outputs
   * \return The total loss from the layer.
   *
   * The Forward wrapper calls the relevant device wrapper function
   * (Forward_cpu or Forward_gpu) to compute the top blob values given the
   * bottom blobs.  If the layer has any non-zero loss_weights, the wrapper
   * then computes and returns the loss.
   *
   * Your layer should implement Forward_cpu and (optionally) Forward_gpu.
   */
  inline Dtype Forward(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

 // 反向传播函数  
 // 输入top和propagate_down  
 // 输出bottom  
  /**
   * @brief Given the top blob error gradients, compute the bottom blob error
   *        gradients.
   *
   * @param top
   *     the output blobs, whose diff fields store the gradient of the error
   *     with respect to themselves
   * @param propagate_down
   *     a vector with equal length to bottom, with each index indicating
   *     whether to propagate the error gradients down to the bottom blob at
   *     the corresponding index
   * @param bottom
   *     the input blobs, whose diff fields will store the gradient of the error
   *     with respect to themselves after Backward is run
   *
   * The Backward wrapper calls the relevant device wrapper function
   * (Backward_cpu or Backward_gpu) to compute the bottom blob diffs given the
   * top blob diffs.
   *
   * Your layer should implement Backward_cpu and (optionally) Backward_gpu.
   */
  inline void Backward(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom);

// 返回blob指针的容器 
  /**
   * @brief Returns the vector of learnable parameter blobs.
   */
  vector<shared_ptr<Blob<Dtype> > >& blobs() {
    return blobs_;
  }
// 返回层的参数  
  /**
   * @brief Returns the layer parameter.
   */
  const LayerParameter& layer_param() const { return layer_param_; }

  /**
   * @brief Writes the layer parameter to a protocol buffer
   */
  virtual void ToProto(LayerParameter* param, bool write_diff = false);

 // 返回标量的损失（该损失与top blob相关联，给定索引就可获得该损失）
  /**
   * @brief Returns the scalar loss associated with a top blob at a given index.
   */
  inline Dtype loss(const int top_index) const {
    return (loss_.size() > top_index) ? loss_[top_index] : Dtype(0);
  }
  
// 给定索引，设置top blob相关联的损失  
  /**
   * @brief Sets the loss associated with a top blob at a given index.
   */
  inline void set_loss(const int top_index, const Dtype value) {
    if (loss_.size() <= top_index) {
      loss_.resize(top_index + 1, Dtype(0));
    }
    loss_[top_index] = value;
  }

  /**
   * @brief Returns the layer type.
   */
   // 虚函数，而且还是内联的，返回层类型  
  virtual inline const char* type() const { return ""; }

  /**
   * @brief Returns the exact number of bottom blobs required by the layer,
   *        or -1 if no exact number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some exact number of bottom blobs.
   */
   // 虚函数，获得bottom blob的精确个数 
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  /**
   * @brief Returns the minimum number of bottom blobs required by the layer,
   *        or -1 if no minimum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some minimum number of bottom blobs.
   */
   // 虚函数，获得bottom blob的最小个数
  virtual inline int MinBottomBlobs() const { return -1; }
  /**
   * @brief Returns the maximum number of bottom blobs required by the layer,
   *        or -1 if no maximum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some maximum number of bottom blobs.
   */
   // 虚函数，获得bottom blob的最大个数  
  virtual inline int MaxBottomBlobs() const { return -1; }
  /**
   * @brief Returns the exact number of top blobs required by the layer,
   *        or -1 if no exact number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some exact number of top blobs.
   */
    // 虚函数，获得top blob的精确个数  
  virtual inline int ExactNumTopBlobs() const { return -1; }
  /**
   * @brief Returns the minimum number of top blobs required by the layer,
   *        or -1 if no minimum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some minimum number of top blobs.
   */
   // 虚函数，获得top blob的最小个数  
  virtual inline int MinTopBlobs() const { return -1; }
  /**
   * @brief Returns the maximum number of top blobs required by the layer,
   *        or -1 if no maximum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some maximum number of top blobs.
   */
   // 虚函数，获得top blob的最大个数  
  virtual inline int MaxTopBlobs() const { return -1; }
  /**
   * @brief Returns true if the layer requires an equal number of bottom and
   *        top blobs.
   *
   * This method should be overridden to return true if your layer expects an
   * equal number of bottom and top blobs.
   */
   // 虚函数，bottom blob和top blob的个数是否一致  
  virtual inline bool EqualNumBottomTopBlobs() const { return false; }

  /**
   * @brief Return whether "anonymous" top blobs are created automatically
   *        by the layer.
   *
   * If this method returns true, Net::Init will create enough "anonymous" top
   * blobs to fulfill the requirement specified by ExactNumTopBlobs() or
   * MinTopBlobs().
   */
   // 返回当前层是否自动创建匿名top blobs  
   // 如果返回true，表明网络初始化的时候创建了了足够多的匿名top blobs  
   // 来满足ExactNumTopBlobs或者MinTopBlobs所要求的top blobs的个数  
  virtual inline bool AutoTopBlobs() const { return false; }

/* 
AllowforceBackward用来设置是否强制梯度返回，因为有些层其实不需要梯度信息 ，后面两个函数分别查看以及设置是是否需要计算梯度。 
*/    
  /**
   * @brief Return whether to allow force_backward for a given bottom blob
   *        index.
   *
   * If AllowForceBackward(i) == false, we will ignore the force_backward
   * setting and backpropagate to blob i only if it needs gradient information
   * (as is done when force_backward == false).
   */
   // 对于一个给定的bottom blob，返回是否允许强制反传 
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

  /**
   * @brief Specifies whether the layer should compute gradients w.r.t. a
   *        parameter at a particular index given by param_id.
   *
   * You can safely ignore false values and always compute gradients
   * for all parameters, but possibly with wasteful computation.
   */
  inline bool param_propagate_down(const int param_id) {
    return (param_propagate_down_.size() > param_id) ?
        param_propagate_down_[param_id] : false;
  }
  //set_param_propagate_down，param_propagate_down 函数：设置对于那些bottom 需要反向传播。 
  /**
   * @brief Sets whether the layer should compute gradients w.r.t. a
   *        parameter at a particular index given by param_id.
   */
  inline void set_param_propagate_down(const int param_id, const bool value) {
    if (param_propagate_down_.size() <= param_id) {
      param_propagate_down_.resize(param_id + 1, true);
    }
    param_propagate_down_[param_id] = value;
  }


 protected:

    // layer中有这三个主要参数：    

    // 层说明参数，从protocal buffers格式的网络结构说明文件中读取  
    // 层的参数 这个是protobuf文件中存储的layer参数
  /** The protobuf that stores the layer parameters */
  LayerParameter layer_param_;

// 层状态，参与网络的训练还是测试  
    // 训练还是测试  
  /** The phase: TRAIN or TEST */
  Phase phase_;

// 层权值和偏置参数，使用向量是因为权值参数和偏置是分开保存在两个blob中的  
    // blobs_的是blob指针容器
    // 这个存储的是layer的参数，在程序中用的 
  /** The vector that stores the learnable parameters as a set of blobs. */
  vector<shared_ptr<Blob<Dtype> > > blobs_; 

  // 标志每个top blob是否需要计算反向传递的梯度值  
    // 是否需要计算梯度，也即是否需要往下传播  
    // 这个bool表示是否计算各个blob参数的diff，即传播误差
  /** Vector indicating whether to compute the diff of each param blob. */
  vector<bool> param_propagate_down_;

// 非LossLayer为零，LossLayer中表示每个top blob计算的loss的权重 
// 每个top blob在目标函数中有非零的权重
  /** The vector that indicates whether each top blob has a non-zero weight in
   *  the objective function. */
  vector<Dtype> loss_;


/////////////////////////////这两个函数非虚函数，它们内部会调用如下虚函数完成数据前向传递和  
/////////////////////////////误差反向传播，根据执行环境的不同每个子类Layer必须重写CPU和GPU版本， 
  // 纯虚函数，必须要实现前向的CPU计算，需要用户去实现全向传播CPU，也就是说必须要实现CPU的前向传播   
  /** @brief Using the CPU device, compute the layer output. */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) = 0;
  
  // 虚函数，需要用户去实现全向传播GPU，如果实现GPU则运行GPU的代码  
  // 如果没有实现则调用默认的CPU的代码 
  /**
   * @brief Using the GPU device, compute the layer output.
   *        Fall back to Forward_cpu() if unavailable.
   */
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    // LOG(WARNING) << "Using CPU code as backup.";
    return Forward_cpu(bottom, top);
  }

  // 纯虚函数，反传CPU ，必须要实现！！ 
  /**
   * @brief Using the CPU device, compute the gradients for any parameters and
   *        for the bottom blobs if propagate_down is true.
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) = 0;

   // 虚函数，反传GPU，如果没有则用CPU的反传 
  /**
   * @brief Using the GPU device, compute the gradients for any parameters and
   *        for the bottom blobs if propagate_down is true.
   *        Fall back to Backward_cpu() if unavailable.
   */
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {
    // LOG(WARNING) << "Using CPU code as backup.";
    Backward_cpu(top, propagate_down, bottom);
  }

  /**
   * Called by the parent Layer's SetUp to check that the number of bottom
   * and top Blobs provided as input match the expected numbers specified by
   * the {ExactNum,Min,Max}{Bottom,Top}Blobs() functions. */
   // 该函数在SetUp中被调用  
   // 检查Blob的一些参数是否正确  
   // 比如:  
   // 精确的底层blob数目  
   // 最小的底层blob数目  
   // 最大的底层blob数目  
   // 精确的顶层blob数目  
   // 最小的顶层blob数目  
   // 最大的顶层blob数目  
   // 此外还检查顶层和底层是否一致  
  virtual void CheckBlobCounts(const vector<Blob<Dtype>*>& bottom,
                               const vector<Blob<Dtype>*>& top) {
    if (ExactNumBottomBlobs() >= 0) {
        // 保证输入bottom 数量和要求的相同  
      CHECK_EQ(ExactNumBottomBlobs(), bottom.size())
          << type() << " Layer takes " << ExactNumBottomBlobs()
          << " bottom blob(s) as input.";
    }
    if (MinBottomBlobs() >= 0) {
        //保证输入的bottom数量大于或等于要求的最小数量  
      CHECK_LE(MinBottomBlobs(), bottom.size())
          << type() << " Layer takes at least " << MinBottomBlobs()
          << " bottom blob(s) as input.";
    }
    if (MaxBottomBlobs() >= 0) {
        //保证输入的bottom数量小于或等于要求的最大数量  
      CHECK_GE(MaxBottomBlobs(), bottom.size())
          << type() << " Layer takes at most " << MaxBottomBlobs()
          << " bottom blob(s) as input.";
    }
    if (ExactNumTopBlobs() >= 0) {
        // 保证输入top数量和要求的相同  
      CHECK_EQ(ExactNumTopBlobs(), top.size())
          << type() << " Layer produces " << ExactNumTopBlobs()
          << " top blob(s) as output.";
    }
    if (MinTopBlobs() >= 0) {
        //保证输入的top数量大于或等于要求的最小数量  
      CHECK_LE(MinTopBlobs(), top.size())
          << type() << " Layer produces at least " << MinTopBlobs()
          << " top blob(s) as output.";
    }
    if (MaxTopBlobs() >= 0) {
        //保证输入的top数量小于或等于要求的最大数量  
      CHECK_GE(MaxTopBlobs(), top.size())
          << type() << " Layer produces at most " << MaxTopBlobs()
          << " top blob(s) as output.";
    }
    if (EqualNumBottomTopBlobs()) {
        //保证输入的bottom数量和输出的top数量相同  
      CHECK_EQ(bottom.size(), top.size())
          << type() << " Layer produces one top blob as output for each "
          << "bottom blob input.";
    }
  }
/* 
SetLoss是非常重要的一个步骤，是被SetUp调用来初始化top bottom的weights，并且存储非零的loss weights 在diff blob里面 
*/  
/*
其中的一些函数的具体实现如下：
主要就是前传和反传，前传调用对应的Forward_cpu或者Forward_gpu
而我们知道Forward_cpu是纯虚函数，必须要实现而Forward_gpu是虚函数，如果不实现就调用 Forward_cpu函数了。
前传（你必须实现自己的Forward_cpu，实现Forward_gpu是可选的）
*/
/** 
   * Called by SetUp to initialize the weights associated with any top blobs in 
   * the loss function. Store non-zero loss weights in the diff blob. 
   * 初始化损失权重---<strong>为每个top blob设置loss weight multiplier blobs(损失权重乘子blobs)</strong>，非LossLayer的top blob的loss weight值为零 
   * <strong>=====!!!! Store non-zero loss weights in the diff blob !!!!=====</strong> 
   */  

  inline void SetLossWeights(const vector<Blob<Dtype>*>& top) {
  //message Layerparameter中的<code>repeated float loss_weight = 5;</code>表示的是“The amount of weight to assign each top blob in the objective”</strong></em>  
    const int num_loss_weights = layer_param_.loss_weight_size();
    if (num_loss_weights) {
      CHECK_EQ(top.size(), num_loss_weights) << "loss_weight must be "
          "unspecified or specified once per top blob.";
      for (int top_id = 0; top_id < top.size(); ++top_id) {
        const Dtype loss_weight = layer_param_.loss_weight(top_id);
        if (loss_weight == Dtype(0)) { continue; }

        //修改Layer的数据成员loss_,其存储的是loss_weight</em>  
        this->set_loss(top_id, loss_weight);
        const int count = top[top_id]->count();

        //返回指向某块Blob的diff所对应的内存空间的指针，并且由于mutable_cpu_diff返回的是void*指针，so，还有一个类型转换过程  
        Dtype* loss_multiplier = top[top_id]->mutable_cpu_diff();

        //loss_multiplier是一个void指针，caffe_set函数表示用loss_weight初始化这块内存，<span style="font-family: Arial, Helvetica, sans-serif;">使其能够存储count个loss_weight(when loss_weight!=0),if loss_weight=0,则用0值来初始化.-----这里为blob的每个元素都初始化了一个loss_weight, 那么在后面计算loss时，只要sum(top)就可以了（我觉得是这样，后面的代码还没看）
        caffe_set(count, loss_weight, loss_multiplier);
      }
    }
  }

 private:
    // 判断该层是否被其他层所共享  
// 这个内部变量实际是判断该层是不是数据层、数据层才可以被其他的网络共享
  /** Whether this layer is actually shared by other nets*/
  bool is_shared_;

// 前向传播的时候所使用的互斥量的指针 
  /** The mutex for sequential forward if this layer is shared */
  shared_ptr<boost::mutex> forward_mutex_;

  /** Initialize forward_mutex_ */
  void InitMutex();

  // 如果该层是共享的，则需要锁住互斥量  
  /** Lock forward_mutex_ if this layer is shared */
  void Lock();

  // 如果该层是共享的，则需要解锁互斥量
  /** Unlock forward_mutex_ if this layer is shared */
  void Unlock();

  DISABLE_COPY_AND_ASSIGN(Layer);
};  // class Layer
// Forward and backward wrappers. You should implement the cpu and  
// gpu specific implementations instead, and should not change these  
// functions.  
// 有一点需要记住的是：在模板类Layer的forward函数里面，会再次调用调用Reshape()函数，也就是说，即使我们每次迭代每个minibatch里的图像（或者特征）的shape不一致，也没有关系，  
// 因为在真正调用forward_cpu / forward_gpu 之前都会重新Reshape；SetUp里面的Reshape只是设置了初始的Top blobs 的shape  

/* 
前传调用对应的Forward_cpu或者Forward_gpu而我们知道Forward_cpu是纯虚函数，必须要实而Forward_gpu是虚函数，如果不实现就调用 Forward_cpu函数了。前传（你必须实现自己的Forward_cpu，实现Forward_gpu是可选的） 
*/  
template <typename Dtype>  
inline Dtype Layer<Dtype>::Forward(const vector<Blob<Dtype>*>& bottom,  
    const vector<Blob<Dtype>*>& top) {  
  // Lock during forward to ensure sequential forward  
  // 前传的时候需要上锁，按照顺序执行才行，否则就乱了  
  Lock();  
  Dtype loss = 0;  
  // 根据bottom设置top的形状  
  Reshape(bottom, top);  
  // 设置运行模式CPU or GPU  
  switch (Caffe::mode()) {  
  case Caffe::CPU:  
    // 调用CPU的前传  
    Forward_cpu(bottom, top);  
    // 前传计算完之后计算损失（只有最后一层才进行计算，其余层都不用）  
    for (int top_id = 0; top_id < top.size(); ++top_id) {  
      if (!this->loss(top_id)) { continue; }  
      const int count = top[top_id]->count();  
      // 获取前传的数据  
      const Dtype* data = top[top_id]->cpu_data();  
      // 获取梯度（\frac{\partial Loss}{\partial net}）  
      const Dtype* loss_weights = top[top_id]->cpu_diff(); 

      //这里的loss_weights我觉得应该是SetLossWeights()方法中模板函数caffe_set()所初始化的loss_weight  
      // data与loss_weight的点积，即得损失函数关于当前层权重的偏导了  
    // \frac{\partial Loss}{\partial net} * \frac{\partial net}{\frac{W}}  
    // = \frac{\partial Loss}{\partial W}  
      loss += caffe_cpu_dot(count, data, loss_weights);  
    }  
    break;  
  case Caffe::GPU:  
    // GPU前传  
    Forward_gpu(bottom, top);  
#ifndef CPU_ONLY  
    // 同上，只不过这里用GPU来计算点积了  
    for (int top_id = 0; top_id < top.size(); ++top_id) {  
      if (!this->loss(top_id)) { continue; }  
      const int count = top[top_id]->count();  
      // 获取GPU上的数据  
      const Dtype* data = top[top_id]->gpu_data();  
      const Dtype* loss_weights = top[top_id]->gpu_diff();  
      Dtype blob_loss = 0;  
      caffe_gpu_dot(count, data, loss_weights, &blob_loss);  
      loss += blob_loss;  
    }  
#endif
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
  Unlock();
  return loss;
}

// 反传的道理与前传的道理很类似
// 反传 ，必须实现CPU，但是GPU是可选的  
template <typename Dtype>
inline void Layer<Dtype>::Backward(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  switch (Caffe::mode()) {  
  case Caffe::CPU:// CPU反传  
  //根据blob top 的error 梯度（diff）计算bottom 的 error 梯度。 propagate_down 是长度   
//和bottom 相同的vector ，用于控制是否需要对对应的bottom 元素传播梯度。具体layer具体定义。 
    Backward_cpu(top, propagate_down, bottom);  
    break;  
  case Caffe::GPU:// GPU反传  
    Backward_gpu(top, propagate_down, bottom);  
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
}

////////////////Layer的序列化函数,将layer的层说明参数layer_param_，层权值和偏置  
////////////////参数blobs_复制到LayerParameter对象，便于写到磁盘，  
// 将LayerParameter转换为ProtoBuf 
// Serialize LayerParameter to protocol buffer
template <typename Dtype>
void Layer<Dtype>::ToProto(LayerParameter* param, bool write_diff) {
  param->Clear();

  // 复制层说明参数layer_param_  
  param->CopyFrom(layer_param_);
  param->clear_blobs();
  for (int i = 0; i < blobs_.size(); ++i) {
      // 复制层权值和偏置参数blobs_  
    //调用Blob的ToProto方法。param->add_blobs()返回Blobproto*,从而将Blob的shape_,data_,diff_分别copy到BlobProto的shape,data,diff,完成序列化 
    blobs_[i]->ToProto(param->add_blobs(), write_diff);
  }
}

}  // namespace caffe

#endif  // CAFFE_LAYER_H_
