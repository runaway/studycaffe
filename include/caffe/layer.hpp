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
��Ҫ������һ��ģ����Layer
���ȣ���һ�����ݳ�Ա����Ҫ�У�
protected��
LayerParameter layer_param_ �� The protobuf that stores the layer parameters����caffe.proto�ļ��ﶨ���message,��Ӧ��caffe.pb.h�ﶨ���һ���ࡣ
Phase phase_  ��The phase: TRAIN or TEST����Phase��caffe.pb.h�ﶨ���һ��ö������
vector<shared_ptr<Blob<Dtype> > > blobs_ ��The vector that stores the learnable parameters as a set of blobs���������������blobs_��洢����ָ��Blob<Dtyp>������ָ�룬Blob<Dtype>����洢����learnable parameter, ʹ����������Ϊweight��bias�Ƿֿ�����������blob�С�
vector<bool> param_propagate_down_ �� Vector indicating whether to compute the diff of each param blob���������Ƿ�Ϊparam blob�����ݶ�diff����־ÿ��top blob�Ƿ���Ҫ���㷴�򴫵ݵ��ݶ�ֵ��
vector<Dtype> loss_ �� The vector that indicates whether each top blob has a non-zero weight in the objective function��������ÿ��top blob �� objective function�Ƿ�non-zero weigh����Losslayer�б�ʾÿ��top blob�����loss��Ȩ�ء�
private��
bool is_shared_ �� Whether this layer is actually shared by other nets
shared_ptr<boost::mutex> forward_mutex_ �� The mutex���������� for sequential forward if this layer is shared
Ȼ��һ�³�Ա������
*/
// ����layer.hpp�ǳ�������Ļ��࣬����������������ϵļ̳У�Ҳ��ʣ�µ����ͷ�ļ�����ͼ�е�������֡�

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
    // Layer��Ĺ�������explicit Layer(const LayerParameter& param) : layer_param_(param)�᳢�Դ�protobuf�ļ���ȡ������
    // ��������Ҫ�ӿڣ�SetUp Forward Backward
    /*
SetUp������Ҫ����ʵ�ʵĲ������ý���ʵ�֣��Ը������͵Ĳ�����ʼ����Forward��Backward��Ӧǰ�����ͷ�����£�����ͳһ����bottom�����Ϊtop������Backward�����и�propagate_down������������ʾ��Layer�Ƿ��򴫲�������

��Forward��Backward�ľ���ʵ��������Caffe::mode()���ж�Ӧ�Ĳ�������ʹ��cpu����gpu���м��㣬������ʵ���˶�Ӧ�Ľӿ�Forward_cpu��Forward_gpu��Backward_cpu��Backward_gpu����Щ�ӿڶ���virtual�����廹��Ҫ����layer�����ͽ��ж�Ӧ�ļ��㣨ע�⣺��Щlayer��û��GPU�����ʵ�֣����Է�װʱ������CPU�ļ�����Ϊ�󱸣������⣬��ʵ����ToProto�Ľӿڣ���Layer�Ĳ���д�뵽protocol buffer�ļ��С�
    */

// �ô�protobuf ����message LayerParameter �е�blobs ��ʼ�� blobs_ 
// blobs_���壺vector<shared_ptr<Blob<Dtype> > > blobs_

/* 
���Ȼ�õ�ǰ�����Phase����train����test���ڳ�ʼ���б��ʼ��LayerParameter,֮��blobs_�����ŵ���һ��ָ��blob���shared_ptrָ���һ��vector��������������ռ䣬Ȼ�󽫴����layer_param�е�blob���������� 
*/  
  // ��ʽ�Ĺ��캯������Ҫ��д���κγ�ʼ������SetUp()�����;���췽��ֻ�ǻ�ȡphaseֵ�����������˵������(layer_param_)���ṩ��Ȩֵ��ƫ�ò�����Ҳ���ơ�    
  /**
   * You should not implement your own constructor. Any set up code should go
   * to SetUp(), where the dimensions of the bottom blobs are provided to the
   * layer.
   */
  explicit Layer(const LayerParameter& param)
    : layer_param_(param), is_shared_(false) {
      // Set phase and copy blobs (if there are any).
      // ѵ�����ǲ��ԣ�phase  
      phase_ = param.phase();

      // ��message Layerparameter�У�<code>repeated BlobProto blobs</code>��ʾ����"The blobs containing the numeric parameters of the layer",  
      // Ҳ����˵����Layer�У�blob�洢���ǲ���numeric parameters������Ȼ����Ҳ��������һ�������ˣ��Ͼ�Blob�������洢���ݵģ���Layer��input bottom blob�Լ�output top blob �����ŵĲ�������ͨ����˵���������ݡ�
      if (layer_param_.blobs_size() > 0) {
        // ��blobs_�Ĵ�С����Ϊ�����еĴ�С  
        blobs_.resize(layer_param_.blobs_size());
        for (int i = 0; i < layer_param_.blobs_size(); ++i) {
          //blobs_��Ԫ����ָ��Blob<Dtype>������ָ��,��Ҫע�������������õ��ǳ�Ա���������һ�����ʹ�õ��Ǽ�ͷ�������reset����Ϊ��������Dtype���ܻᷢ���仯  
          // �½����ɸ�Blob  
          blobs_[i].reset(new Blob<Dtype>());

          //���õ���Blob���͵�FromProto����    
          // ��blob�ļ��л�ȡ����  
          blobs_[i]->FromProto(layer_param_.blobs(i));
        } //��ȡ����Ȩֵ��ƫ�ò���  
      }
    }
  virtual ~Layer() {}

/** 
   * @brief Implements common layer setup functionality. 
   *        ʵ��ÿ��layer�����setup���� 
   * 
   * @param bottom the preshaped input blobs 
   *        ����������ݣ�blob�еĴ洢�ռ������� 
   * @param top 
   *     the allocated but unshaped output blobs, to be shaped by Reshape 
   *     ���������ݣ�blob�����ѹ��쵫�����еĴ洢�ռ�δ���룬������Reshape������ʵ�� 
   * 
   * Checks that the number of bottom and top blobs is correct. 
   * Calls LayerSetUp to do special layer setup for individual layer types, 
   * followed by Reshape to set up sizes of top blobs and internal buffers. 
   * S<strong>ets up the loss weight multiplier blobs for any non-zero loss weights</strong>. 
   * This method may not be overridden. 
   * ��ʼ�����캯��SetUp 
   * 1. ����������blob�����Ƿ�����Ҫ��ÿ�����ܴ��������������ݲ�һ�� 
   * 2. ����LayerSetUp������ʼ������Ĳ㣬ÿ��Layer��������д���������ɶ��Ƶĳ�ʼ�� 
   * 3. ����Reshape����Ϊtop blob������ʴ�С�Ĵ洢�ռ� 
   * 4. Ϊÿ��top blob����loss weight multiplier blobs(��ʧȨ�س���blobs)����LossLayer��top blob��loss weightֵΪ��.<strong>---!!!Sets up the loss weight multiplier blobs for any non-zero loss weights!!!---</strong> 
   * 
   * �˷������麯����������д��ģʽ�̶� 
   */  

// �麯����������ض�layer�����ࣩ
// SetUp���ò�Ļ����������BLOB�Ĳ���������LayerSetUp���г�ʼ��  
// LayerSetUp��һ���麯�����û�����ȥ��������  
// Ȼ��������topblob����״�Լ�������ʧȨ�ء� 

  void SetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
      // ��ʼ��������  
    InitMutex();
      // ���Blob 
    CheckBlobCounts(bottom, top);
      // ��ĳ�ʼ�����麯�������û�ȥʵ����γ�ʼ���㣩  
    LayerSetUp(bottom, top);
      // �ı�top����״���麯�������û�ȥʵ����θ���bottomblob�ı�topblob����״��  
    Reshape(bottom, top);
       // ������ʧȨ��  
    SetLossWeights(top);
  }
/** 
   * @brief Does layer-specific setup: your layer should implement this function 
   *        as well as Reshape. 
   *        ���Ƴ�ʼ����ÿ������layer����ʵ�ִ��麯�������� 
   * 
   * @param bottom 
   *     the preshaped input blobs, whose data fields store the input data for 
   *     this layer 
   *     ����blob, ���ݳ�Աdata_��diff_�洢��������� 
   * @param top 
   *     the allocated but unshaped output blobs 
   *     ���blob, blob�����ѹ��쵫���ݳ�Ա�Ŀռ���δ���� 
   * 
   * This method should do one-time layer specific setup. This includes reading 
   * and processing relevent parameters from the <code>layer_param_</code>. 
   * Setting up the shapes of top blobs and internal buffers should be done in 
   * <code>Reshape</code>, which will be called before the forward pass to 
   * adjust the top blob sizes. 
   * �˷���ִ��һ�ζ��ƻ��Ĳ��ʼ����������layer_param_���벢������صĲ�Ȩֵ��ƫ�ò����� 
   * ����Reshape��������top blob�Ĵ洢�ռ� 
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

// �жϸò��Ƿ�������ģʽ�����Ƿ����ݲ��л��ˣ� 
  /** @brief Return whether this layer is actually shared by other nets.
   *         If ShareInParallel() is true and using more than one GPU and the
   *         net has TRAIN phase, then this function is expected return true.
   */
  inline bool IsShared() const { return is_shared_; }

 // �����Ƿ��� 
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
   * <strong>-----reshape top blobs �Լ� internal buffers����Ӧbottom (input) blobs-----bottom��top���ж��blob</strong> 
   */  

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) = 0;

// ǰ�򴫲�����  
// ����bottom�������top  
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

 // ���򴫲�����  
 // ����top��propagate_down  
 // ���bottom  
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

// ����blobָ������� 
  /**
   * @brief Returns the vector of learnable parameter blobs.
   */
  vector<shared_ptr<Blob<Dtype> > >& blobs() {
    return blobs_;
  }
// ���ز�Ĳ���  
  /**
   * @brief Returns the layer parameter.
   */
  const LayerParameter& layer_param() const { return layer_param_; }

  /**
   * @brief Writes the layer parameter to a protocol buffer
   */
  virtual void ToProto(LayerParameter* param, bool write_diff = false);

 // ���ر�������ʧ������ʧ��top blob����������������Ϳɻ�ø���ʧ��
  /**
   * @brief Returns the scalar loss associated with a top blob at a given index.
   */
  inline Dtype loss(const int top_index) const {
    return (loss_.size() > top_index) ? loss_[top_index] : Dtype(0);
  }
  
// ��������������top blob���������ʧ  
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
   // �麯�������һ��������ģ����ز�����  
  virtual inline const char* type() const { return ""; }

  /**
   * @brief Returns the exact number of bottom blobs required by the layer,
   *        or -1 if no exact number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some exact number of bottom blobs.
   */
   // �麯�������bottom blob�ľ�ȷ���� 
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  /**
   * @brief Returns the minimum number of bottom blobs required by the layer,
   *        or -1 if no minimum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some minimum number of bottom blobs.
   */
   // �麯�������bottom blob����С����
  virtual inline int MinBottomBlobs() const { return -1; }
  /**
   * @brief Returns the maximum number of bottom blobs required by the layer,
   *        or -1 if no maximum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some maximum number of bottom blobs.
   */
   // �麯�������bottom blob��������  
  virtual inline int MaxBottomBlobs() const { return -1; }
  /**
   * @brief Returns the exact number of top blobs required by the layer,
   *        or -1 if no exact number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some exact number of top blobs.
   */
    // �麯�������top blob�ľ�ȷ����  
  virtual inline int ExactNumTopBlobs() const { return -1; }
  /**
   * @brief Returns the minimum number of top blobs required by the layer,
   *        or -1 if no minimum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some minimum number of top blobs.
   */
   // �麯�������top blob����С����  
  virtual inline int MinTopBlobs() const { return -1; }
  /**
   * @brief Returns the maximum number of top blobs required by the layer,
   *        or -1 if no maximum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some maximum number of top blobs.
   */
   // �麯�������top blob��������  
  virtual inline int MaxTopBlobs() const { return -1; }
  /**
   * @brief Returns true if the layer requires an equal number of bottom and
   *        top blobs.
   *
   * This method should be overridden to return true if your layer expects an
   * equal number of bottom and top blobs.
   */
   // �麯����bottom blob��top blob�ĸ����Ƿ�һ��  
  virtual inline bool EqualNumBottomTopBlobs() const { return false; }

  /**
   * @brief Return whether "anonymous" top blobs are created automatically
   *        by the layer.
   *
   * If this method returns true, Net::Init will create enough "anonymous" top
   * blobs to fulfill the requirement specified by ExactNumTopBlobs() or
   * MinTopBlobs().
   */
   // ���ص�ǰ���Ƿ��Զ���������top blobs  
   // �������true�����������ʼ����ʱ�򴴽������㹻�������top blobs  
   // ������ExactNumTopBlobs����MinTopBlobs��Ҫ���top blobs�ĸ���  
  virtual inline bool AutoTopBlobs() const { return false; }

/* 
AllowforceBackward���������Ƿ�ǿ���ݶȷ��أ���Ϊ��Щ����ʵ����Ҫ�ݶ���Ϣ ���������������ֱ�鿴�Լ��������Ƿ���Ҫ�����ݶȡ� 
*/    
  /**
   * @brief Return whether to allow force_backward for a given bottom blob
   *        index.
   *
   * If AllowForceBackward(i) == false, we will ignore the force_backward
   * setting and backpropagate to blob i only if it needs gradient information
   * (as is done when force_backward == false).
   */
   // ����һ��������bottom blob�������Ƿ�����ǿ�Ʒ��� 
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
  //set_param_propagate_down��param_propagate_down ���������ö�����Щbottom ��Ҫ���򴫲��� 
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

    // layer������������Ҫ������    

    // ��˵����������protocal buffers��ʽ������ṹ˵���ļ��ж�ȡ  
    // ��Ĳ��� �����protobuf�ļ��д洢��layer����
  /** The protobuf that stores the layer parameters */
  LayerParameter layer_param_;

// ��״̬�����������ѵ�����ǲ���  
    // ѵ�����ǲ���  
  /** The phase: TRAIN or TEST */
  Phase phase_;

// ��Ȩֵ��ƫ�ò�����ʹ����������ΪȨֵ������ƫ���Ƿֿ�����������blob�е�  
    // blobs_����blobָ������
    // ����洢����layer�Ĳ������ڳ������õ� 
  /** The vector that stores the learnable parameters as a set of blobs. */
  vector<shared_ptr<Blob<Dtype> > > blobs_; 

  // ��־ÿ��top blob�Ƿ���Ҫ���㷴�򴫵ݵ��ݶ�ֵ  
    // �Ƿ���Ҫ�����ݶȣ�Ҳ���Ƿ���Ҫ���´���  
    // ���bool��ʾ�Ƿ�������blob������diff�����������
  /** Vector indicating whether to compute the diff of each param blob. */
  vector<bool> param_propagate_down_;

// ��LossLayerΪ�㣬LossLayer�б�ʾÿ��top blob�����loss��Ȩ�� 
// ÿ��top blob��Ŀ�꺯�����з����Ȩ��
  /** The vector that indicates whether each top blob has a non-zero weight in
   *  the objective function. */
  vector<Dtype> loss_;


/////////////////////////////�������������麯���������ڲ�����������麯���������ǰ�򴫵ݺ�  
/////////////////////////////���򴫲�������ִ�л����Ĳ�ͬÿ������Layer������дCPU��GPU�汾�� 
  // ���麯��������Ҫʵ��ǰ���CPU���㣬��Ҫ�û�ȥʵ��ȫ�򴫲�CPU��Ҳ����˵����Ҫʵ��CPU��ǰ�򴫲�   
  /** @brief Using the CPU device, compute the layer output. */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) = 0;
  
  // �麯������Ҫ�û�ȥʵ��ȫ�򴫲�GPU�����ʵ��GPU������GPU�Ĵ���  
  // ���û��ʵ�������Ĭ�ϵ�CPU�Ĵ��� 
  /**
   * @brief Using the GPU device, compute the layer output.
   *        Fall back to Forward_cpu() if unavailable.
   */
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    // LOG(WARNING) << "Using CPU code as backup.";
    return Forward_cpu(bottom, top);
  }

  // ���麯��������CPU ������Ҫʵ�֣��� 
  /**
   * @brief Using the CPU device, compute the gradients for any parameters and
   *        for the bottom blobs if propagate_down is true.
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) = 0;

   // �麯��������GPU�����û������CPU�ķ��� 
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
   // �ú�����SetUp�б�����  
   // ���Blob��һЩ�����Ƿ���ȷ  
   // ����:  
   // ��ȷ�ĵײ�blob��Ŀ  
   // ��С�ĵײ�blob��Ŀ  
   // ���ĵײ�blob��Ŀ  
   // ��ȷ�Ķ���blob��Ŀ  
   // ��С�Ķ���blob��Ŀ  
   // ���Ķ���blob��Ŀ  
   // ���⻹��鶥��͵ײ��Ƿ�һ��  
  virtual void CheckBlobCounts(const vector<Blob<Dtype>*>& bottom,
                               const vector<Blob<Dtype>*>& top) {
    if (ExactNumBottomBlobs() >= 0) {
        // ��֤����bottom ������Ҫ�����ͬ  
      CHECK_EQ(ExactNumBottomBlobs(), bottom.size())
          << type() << " Layer takes " << ExactNumBottomBlobs()
          << " bottom blob(s) as input.";
    }
    if (MinBottomBlobs() >= 0) {
        //��֤�����bottom�������ڻ����Ҫ�����С����  
      CHECK_LE(MinBottomBlobs(), bottom.size())
          << type() << " Layer takes at least " << MinBottomBlobs()
          << " bottom blob(s) as input.";
    }
    if (MaxBottomBlobs() >= 0) {
        //��֤�����bottom����С�ڻ����Ҫ����������  
      CHECK_GE(MaxBottomBlobs(), bottom.size())
          << type() << " Layer takes at most " << MaxBottomBlobs()
          << " bottom blob(s) as input.";
    }
    if (ExactNumTopBlobs() >= 0) {
        // ��֤����top������Ҫ�����ͬ  
      CHECK_EQ(ExactNumTopBlobs(), top.size())
          << type() << " Layer produces " << ExactNumTopBlobs()
          << " top blob(s) as output.";
    }
    if (MinTopBlobs() >= 0) {
        //��֤�����top�������ڻ����Ҫ�����С����  
      CHECK_LE(MinTopBlobs(), top.size())
          << type() << " Layer produces at least " << MinTopBlobs()
          << " top blob(s) as output.";
    }
    if (MaxTopBlobs() >= 0) {
        //��֤�����top����С�ڻ����Ҫ����������  
      CHECK_GE(MaxTopBlobs(), top.size())
          << type() << " Layer produces at most " << MaxTopBlobs()
          << " top blob(s) as output.";
    }
    if (EqualNumBottomTopBlobs()) {
        //��֤�����bottom�����������top������ͬ  
      CHECK_EQ(bottom.size(), top.size())
          << type() << " Layer produces one top blob as output for each "
          << "bottom blob input.";
    }
  }
/* 
SetLoss�Ƿǳ���Ҫ��һ�����裬�Ǳ�SetUp��������ʼ��top bottom��weights�����Ҵ洢�����loss weights ��diff blob���� 
*/  
/*
���е�һЩ�����ľ���ʵ�����£�
��Ҫ����ǰ���ͷ�����ǰ�����ö�Ӧ��Forward_cpu����Forward_gpu
������֪��Forward_cpu�Ǵ��麯��������Ҫʵ�ֶ�Forward_gpu���麯���������ʵ�־͵��� Forward_cpu�����ˡ�
ǰ���������ʵ���Լ���Forward_cpu��ʵ��Forward_gpu�ǿ�ѡ�ģ�
*/
/** 
   * Called by SetUp to initialize the weights associated with any top blobs in 
   * the loss function. Store non-zero loss weights in the diff blob. 
   * ��ʼ����ʧȨ��---<strong>Ϊÿ��top blob����loss weight multiplier blobs(��ʧȨ�س���blobs)</strong>����LossLayer��top blob��loss weightֵΪ�� 
   * <strong>=====!!!! Store non-zero loss weights in the diff blob !!!!=====</strong> 
   */  

  inline void SetLossWeights(const vector<Blob<Dtype>*>& top) {
  //message Layerparameter�е�<code>repeated float loss_weight = 5;</code>��ʾ���ǡ�The amount of weight to assign each top blob in the objective��</strong></em>  
    const int num_loss_weights = layer_param_.loss_weight_size();
    if (num_loss_weights) {
      CHECK_EQ(top.size(), num_loss_weights) << "loss_weight must be "
          "unspecified or specified once per top blob.";
      for (int top_id = 0; top_id < top.size(); ++top_id) {
        const Dtype loss_weight = layer_param_.loss_weight(top_id);
        if (loss_weight == Dtype(0)) { continue; }

        //�޸�Layer�����ݳ�Աloss_,��洢����loss_weight</em>  
        this->set_loss(top_id, loss_weight);
        const int count = top[top_id]->count();

        //����ָ��ĳ��Blob��diff����Ӧ���ڴ�ռ��ָ�룬��������mutable_cpu_diff���ص���void*ָ�룬so������һ������ת������  
        Dtype* loss_multiplier = top[top_id]->mutable_cpu_diff();

        //loss_multiplier��һ��voidָ�룬caffe_set������ʾ��loss_weight��ʼ������ڴ棬<span style="font-family: Arial, Helvetica, sans-serif;">ʹ���ܹ��洢count��loss_weight(when loss_weight!=0),if loss_weight=0,����0ֵ����ʼ��.-----����Ϊblob��ÿ��Ԫ�ض���ʼ����һ��loss_weight, ��ô�ں������lossʱ��ֻҪsum(top)�Ϳ����ˣ��Ҿ���������������Ĵ��뻹û����
        caffe_set(count, loss_weight, loss_multiplier);
      }
    }
  }

 private:
    // �жϸò��Ƿ�������������  
// ����ڲ�����ʵ�����жϸò��ǲ������ݲ㡢���ݲ�ſ��Ա����������繲��
  /** Whether this layer is actually shared by other nets*/
  bool is_shared_;

// ǰ�򴫲���ʱ����ʹ�õĻ�������ָ�� 
  /** The mutex for sequential forward if this layer is shared */
  shared_ptr<boost::mutex> forward_mutex_;

  /** Initialize forward_mutex_ */
  void InitMutex();

  // ����ò��ǹ���ģ�����Ҫ��ס������  
  /** Lock forward_mutex_ if this layer is shared */
  void Lock();

  // ����ò��ǹ���ģ�����Ҫ����������
  /** Unlock forward_mutex_ if this layer is shared */
  void Unlock();

  DISABLE_COPY_AND_ASSIGN(Layer);
};  // class Layer
// Forward and backward wrappers. You should implement the cpu and  
// gpu specific implementations instead, and should not change these  
// functions.  
// ��һ����Ҫ��ס���ǣ���ģ����Layer��forward�������棬���ٴε��õ���Reshape()������Ҳ����˵����ʹ����ÿ�ε���ÿ��minibatch���ͼ�񣨻�����������shape��һ�£�Ҳû�й�ϵ��  
// ��Ϊ����������forward_cpu / forward_gpu ֮ǰ��������Reshape��SetUp�����Reshapeֻ�������˳�ʼ��Top blobs ��shape  

/* 
ǰ�����ö�Ӧ��Forward_cpu����Forward_gpu������֪��Forward_cpu�Ǵ��麯��������Ҫʵ��Forward_gpu���麯���������ʵ�־͵��� Forward_cpu�����ˡ�ǰ���������ʵ���Լ���Forward_cpu��ʵ��Forward_gpu�ǿ�ѡ�ģ� 
*/  
template <typename Dtype>  
inline Dtype Layer<Dtype>::Forward(const vector<Blob<Dtype>*>& bottom,  
    const vector<Blob<Dtype>*>& top) {  
  // Lock during forward to ensure sequential forward  
  // ǰ����ʱ����Ҫ����������˳��ִ�в��У����������  
  Lock();  
  Dtype loss = 0;  
  // ����bottom����top����״  
  Reshape(bottom, top);  
  // ��������ģʽCPU or GPU  
  switch (Caffe::mode()) {  
  case Caffe::CPU:  
    // ����CPU��ǰ��  
    Forward_cpu(bottom, top);  
    // ǰ��������֮�������ʧ��ֻ�����һ��Ž��м��㣬����㶼���ã�  
    for (int top_id = 0; top_id < top.size(); ++top_id) {  
      if (!this->loss(top_id)) { continue; }  
      const int count = top[top_id]->count();  
      // ��ȡǰ��������  
      const Dtype* data = top[top_id]->cpu_data();  
      // ��ȡ�ݶȣ�\frac{\partial Loss}{\partial net}��  
      const Dtype* loss_weights = top[top_id]->cpu_diff(); 

      //�����loss_weights�Ҿ���Ӧ����SetLossWeights()������ģ�庯��caffe_set()����ʼ����loss_weight  
      // data��loss_weight�ĵ����������ʧ�������ڵ�ǰ��Ȩ�ص�ƫ����  
    // \frac{\partial Loss}{\partial net} * \frac{\partial net}{\frac{W}}  
    // = \frac{\partial Loss}{\partial W}  
      loss += caffe_cpu_dot(count, data, loss_weights);  
    }  
    break;  
  case Caffe::GPU:  
    // GPUǰ��  
    Forward_gpu(bottom, top);  
#ifndef CPU_ONLY  
    // ͬ�ϣ�ֻ����������GPU����������  
    for (int top_id = 0; top_id < top.size(); ++top_id) {  
      if (!this->loss(top_id)) { continue; }  
      const int count = top[top_id]->count();  
      // ��ȡGPU�ϵ�����  
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

// �����ĵ�����ǰ���ĵ��������
// ���� ������ʵ��CPU������GPU�ǿ�ѡ��  
template <typename Dtype>
inline void Layer<Dtype>::Backward(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  switch (Caffe::mode()) {  
  case Caffe::CPU:// CPU����  
  //����blob top ��error �ݶȣ�diff������bottom �� error �ݶȡ� propagate_down �ǳ���   
//��bottom ��ͬ��vector �����ڿ����Ƿ���Ҫ�Զ�Ӧ��bottom Ԫ�ش����ݶȡ�����layer���嶨�塣 
    Backward_cpu(top, propagate_down, bottom);  
    break;  
  case Caffe::GPU:// GPU����  
    Backward_gpu(top, propagate_down, bottom);  
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
}

////////////////Layer�����л�����,��layer�Ĳ�˵������layer_param_����Ȩֵ��ƫ��  
////////////////����blobs_���Ƶ�LayerParameter���󣬱���д�����̣�  
// ��LayerParameterת��ΪProtoBuf 
// Serialize LayerParameter to protocol buffer
template <typename Dtype>
void Layer<Dtype>::ToProto(LayerParameter* param, bool write_diff) {
  param->Clear();

  // ���Ʋ�˵������layer_param_  
  param->CopyFrom(layer_param_);
  param->clear_blobs();
  for (int i = 0; i < blobs_.size(); ++i) {
      // ���Ʋ�Ȩֵ��ƫ�ò���blobs_  
    //����Blob��ToProto������param->add_blobs()����Blobproto*,�Ӷ���Blob��shape_,data_,diff_�ֱ�copy��BlobProto��shape,data,diff,������л� 
    blobs_[i]->ToProto(param->add_blobs(), write_diff);
  }
}

}  // namespace caffe

#endif  // CAFFE_LAYER_H_
