#ifndef CAFFE_SGD_SOLVERS_HPP_
#define CAFFE_SGD_SOLVERS_HPP_

#include <string>
#include <vector>

#include "caffe/solver.hpp"

/*
��ĿǰΪֹ��caffe�ܹ��ṩ�������Ż�������
Stochastic Gradient Descent (type: "SGD"),
AdaDelta (type: "AdaDelta"),
Adaptive Gradient (type: "AdaGrad"),
Adam (type: "Adam"),
Nesterov��s Accelerated Gradient (type: "Nesterov") and
RMSprop (type: "RMSProp")
Solver��������ʹloss��С�����Ż�����������һ�����ݼ�D����Ҫ�Ż���Ŀ�꺯�����������ݼ�����������loss��ƽ��ֵ��

���У�fW(x(i))�����������x(i)�ϵ�loss, �Ƚ�ÿ������������x��loss�������Ȼ����ͣ�������ֵ�� r(W)�������weight_decay)��Ϊ�˼������������
�����������Loss ����������һ����Ҫ�����������ݼ��������ݼ��ǳ����������£����ַ�����Ч�ʺܵͣ����Ҳ��������֪���ݶ��½����õķ�����


��ʵ���У�ͨ�����������ݼ��ֳɼ�����batches), ÿһ������һ��mini-batch����������batch_size)ΪN<<|D|����ʱ��loss ����Ϊ��
 


����loss�����󣬾Ϳ��Ե��������loss���ݶ����Ż�������⡣���������У���forward pass�����loss����backward pass������ݶȡ�
��caffe�У�Ĭ�ϲ��õ�Stochastic Gradient Descent��SGD�������Ż���⡣���漸�ַ���Ҳ�ǻ����ݶȵ��Ż�������like SGD������˱���ֻ����һ��SGD�������ķ���������Ȥ��ͬѧ������ȥ������ԭ�ġ�
*/

namespace caffe {


/*
1��Stochastic gradient descent��SGD)
����ݶ��½���Stochastic gradient descent�������ݶ��½�����gradient descent���Ļ����Ϸ�չ�����ģ��ݶ��½���Ҳ�������½���������ԭ�������׹����Ρ�����ѧϰ���У����������Ѿ�����÷ǳ���ϸ��SGD��ͨ�����ݶȺ���һ�ε�Ȩ�ظ���ֵVt���������������W��������ʽ���£�
 

 
���У�  �Ǹ��ݶȵ�ѧϰ��(base_lr)������һ���ݶ�ֵ��Ȩ�أ�momentum����������Ȩ֮ǰ�ݶȷ���������ݶ��½������Ӱ�졣������������Ҫͨ��tuning���õ���õĽ����һ���Ǹ��ݾ����趨�ġ�����㲻֪������趨��Щ���������Բο���ص����ġ�
�����ѧϰ��ʹ��SGD���ȽϺõĳ�ʼ�������Ĳ����ǰ�ѧϰ����Ϊ0.01���ң�base_lr: 0.01)����ѵ���Ĺ����У����loss��ʼ�����ȶ�ˮƽʱ����ѧϰ�ʳ���һ���������ӣ�gamma���������Ĺ����ظ���Ρ�
����momentum��һ��ȡֵ��0.5--0.99֮�䡣ͨ����Ϊ0.9��momentum������ʹ��SGD�����ѧϰ���������ȶ��Լ����١�
���ڸ����momentum����ο�Hinton�ġ�A Practical Guide to Training Restricted Boltzmann Machines����  
ʵ���� 
[cpp] view plain copy
base_lr: 0.01   
lr_policy: "step"  
gamma: 0.1     
stepsize: 1000    
max_iter: 3500   
momentum: 0.9  
lr_policy����Ϊstep,��ѧϰ�ʵı仯����Ϊ base_lr * gamma ^ (floor(iter / stepsize))
��ǰ1000�ε�����ѧϰ��Ϊ0.01; ��1001-2000�ε�����ѧϰ��Ϊ0.001; ��2001-3000�ε�����ѧϰ��Ϊ0.00001����3001-3500�ε�����ѧϰ��Ϊ10-5  
���������ֻ����Ϊһ��ָ�������ǲ��ܱ�֤���κ�����¶��ܵõ���ѵĽ������ʱ�����ַ���������work�����ѧϰ��ʱ�����diverge�����磬��һ��ʼ�ͷ��ַǳ������NaN����inf��lossֵ�������������ʱ����Ҫ����base_lr��ֵ�����磬0.001����Ȼ������ѵ���������Ĺ����ظ�����ֱ�����ҵ�����work��base_lr��
*/

/**
 * @brief Optimizes the parameters of a Net using
 *        stochastic gradient descent (SGD) with momentum.
 */
template <typename Dtype>
class SGDSolver : public Solver<Dtype> {
 public:
  explicit SGDSolver(const SolverParameter& param)
      : Solver<Dtype>(param) { PreSolve(); }
  explicit SGDSolver(const string& param_file)
      : Solver<Dtype>(param_file) { PreSolve(); }
  virtual inline const char* type() const { return "SGD"; }

  const vector<shared_ptr<Blob<Dtype> > >& history() { return history_; }

 protected:
  void PreSolve();
  Dtype GetLearningRate();
  virtual void ApplyUpdate();
  virtual void Normalize(int param_id);
  virtual void Regularize(int param_id);
  virtual void ComputeUpdateValue(int param_id, Dtype rate);
  virtual void ClipGradients();
  virtual void SnapshotSolverState(const string& model_filename);
  virtual void SnapshotSolverStateToBinaryProto(const string& model_filename);
  virtual void SnapshotSolverStateToHDF5(const string& model_filename);
  virtual void RestoreSolverStateFromHDF5(const string& state_file);
  virtual void RestoreSolverStateFromBinaryProto(const string& state_file);
  // history maintains the historical momentum data.
  // update maintains update related data and is not needed in snapshots.
  // temp maintains other information that might be needed in computation
  //   of gradients/updates and is not needed in snapshots
  vector<shared_ptr<Blob<Dtype> > > history_, update_, temp_;

  DISABLE_COPY_AND_ASSIGN(SGDSolver);
};


/*
5��NAG
Nesterov �ļ����ݶȷ���Nesterov��s accelerated gradient����Ϊ͹�Ż���������ķ������������ٶȷǳ��졣
 ����Ľ������ף�
 I. Sutskever, J. Martens, G. Dahl, and G. Hinton. On the Importance of Initialization and Momentum in Deep Learning. Proceedings of the 30th International Conference on Machine Learning, 2013.
ʾ����
[cpp] view plain copy
net: "examples/mnist/mnist_autoencoder.prototxt"  
test_state: { stage: 'test-on-train' }  
test_iter: 500  
test_state: { stage: 'test-on-test' }  
test_iter: 100  
test_interval: 500  
test_compute_loss: true  
base_lr: 0.01  
lr_policy: "step"  
gamma: 0.1  
stepsize: 10000  
display: 100  
max_iter: 65000  
weight_decay: 0.0005  
snapshot: 10000  
snapshot_prefix: "examples/mnist/mnist_autoencoder_nesterov_train"  
momentum: 0.95  
# solver mode: CPU or GPU  
solver_mode: GPU  
type: "Nesterov"  

*/
template <typename Dtype>
class NesterovSolver : public SGDSolver<Dtype> {
 public:
  explicit NesterovSolver(const SolverParameter& param)
      : SGDSolver<Dtype>(param) {}
  explicit NesterovSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) {}
  virtual inline const char* type() const { return "Nesterov"; }

 protected:
  virtual void ComputeUpdateValue(int param_id, Dtype rate);

  DISABLE_COPY_AND_ASSIGN(NesterovSolver);
};


/*
����Ӧ�ݶȣ�adaptive gradient���ǻ����ݶȵ��Ż�������like SGD��
����Ľ������ף�
Duchi, E. Hazan, and Y. Singer. Adaptive Subgradient Methods for Online Learning and Stochastic Optimization. The Journal of Machine Learning Research, 2011.
ʾ����
[cpp] view plain copy
net: "examples/mnist/mnist_autoencoder.prototxt"  
test_state: { stage: 'test-on-train' }  
test_iter: 500  
test_state: { stage: 'test-on-test' }  
test_iter: 100  
test_interval: 500  
test_compute_loss: true  
base_lr: 0.01  
lr_policy: "fixed"  
display: 100  
max_iter: 65000  
weight_decay: 0.0005  
snapshot: 10000  
snapshot_prefix: "examples/mnist/mnist_autoencoder_adagrad_train"  
# solver mode: CPU or GPU  
solver_mode: GPU  
type: "AdaGrad"  

*/
template <typename Dtype>
class AdaGradSolver : public SGDSolver<Dtype> {
 public:
  explicit AdaGradSolver(const SolverParameter& param)
      : SGDSolver<Dtype>(param) { constructor_sanity_check(); }
  explicit AdaGradSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) { constructor_sanity_check(); }
  virtual inline const char* type() const { return "AdaGrad"; }

 protected:
  virtual void ComputeUpdateValue(int param_id, Dtype rate);
  void constructor_sanity_check() {
    CHECK_EQ(0, this->param_.momentum())
        << "Momentum cannot be used with AdaGrad.";
  }

  DISABLE_COPY_AND_ASSIGN(AdaGradSolver);
};

/*
6��RMSprop
RMSprop��Tieleman��һ�� Coursera�γ��ݽ���������ģ�Ҳ��һ�ֻ����ݶȵ��Ż�������like SGD��
����Ľ������ף�
T. Tieleman, and G. Hinton. RMSProp: Divide the gradient by a running average of its recent magnitude. COURSERA: Neural Networks for Machine Learning.Technical report, 2012.
 ʾ����
[cpp] view plain copy
net: "examples/mnist/lenet_train_test.prototxt"  
test_iter: 100  
test_interval: 500  
base_lr: 1.0  
lr_policy: "fixed"  
momentum: 0.95  
weight_decay: 0.0005  
display: 100  
max_iter: 10000  
snapshot: 5000  
snapshot_prefix: "examples/mnist/lenet_adadelta"  
solver_mode: GPU  
type: "RMSProp"  
rms_decay: 0.98  

������У���Ҫ����rms_decayֵ��
*/
template <typename Dtype>
class RMSPropSolver : public SGDSolver<Dtype> {
 public:
  explicit RMSPropSolver(const SolverParameter& param)
      : SGDSolver<Dtype>(param) { constructor_sanity_check(); }
  explicit RMSPropSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) { constructor_sanity_check(); }
  virtual inline const char* type() const { return "RMSProp"; }

 protected:
  virtual void ComputeUpdateValue(int param_id, Dtype rate);
  void constructor_sanity_check() {
    CHECK_EQ(0, this->param_.momentum())
        << "Momentum cannot be used with RMSProp.";
    CHECK_GE(this->param_.rms_decay(), 0)
        << "rms_decay should lie between 0 and 1.";
    CHECK_LT(this->param_.rms_decay(), 1)
        << "rms_decay should lie between 0 and 1.";
  }

  DISABLE_COPY_AND_ASSIGN(RMSPropSolver);
};


/*
AdaDelta��һ�֡�³����ѧϰ�ʷ��������ǻ����ݶȵ��Ż�������like SGD����
����Ľ������ף�
M. Zeiler ADADELTA: AN ADAPTIVE LEARNING RATE METHOD. arXiv preprint, 2012.
ʾ����
[cpp] view plain copy
net: "examples/mnist/lenet_train_test.prototxt"  
test_iter: 100  
test_interval: 500  
base_lr: 1.0  
lr_policy: "fixed"  
momentum: 0.95  
weight_decay: 0.0005  
display: 100  
max_iter: 10000  
snapshot: 5000  
snapshot_prefix: "examples/mnist/lenet_adadelta"  
solver_mode: GPU  
type: "AdaDelta"  
delta: 1e-6  

��������пɿ���������solver typeΪAdadeltaʱ����Ҫ����delta��ֵ��
*/
template <typename Dtype>
class AdaDeltaSolver : public SGDSolver<Dtype> {
 public:
  explicit AdaDeltaSolver(const SolverParameter& param)
      : SGDSolver<Dtype>(param) { AdaDeltaPreSolve(); }
  explicit AdaDeltaSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) { AdaDeltaPreSolve(); }
  virtual inline const char* type() const { return "AdaDelta"; }

 protected:
  void AdaDeltaPreSolve();
  virtual void ComputeUpdateValue(int param_id, Dtype rate);

  DISABLE_COPY_AND_ASSIGN(AdaDeltaSolver);
};


/*
Adam
��һ�ֻ����ݶȵ��Ż�������like SGD����
 ����Ľ������ף�
D. Kingma, J. Ba. Adam: A Method for Stochastic Optimization. International Conference for Learning Representations, 2015.
*/
/**
 * @brief AdamSolver, an algorithm for first-order gradient-based optimization
 *        of stochastic objective functions, based on adaptive estimates of
 *        lower-order moments. Described in [1].
 *
 * [1] D. P. Kingma and J. L. Ba, "ADAM: A Method for Stochastic Optimization."
 *     arXiv preprint arXiv:1412.6980v8 (2014).
 */
template <typename Dtype>
class AdamSolver : public SGDSolver<Dtype> {
 public:
  explicit AdamSolver(const SolverParameter& param)
      : SGDSolver<Dtype>(param) { AdamPreSolve();}
  explicit AdamSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) { AdamPreSolve(); }
  virtual inline const char* type() const { return "Adam"; }

 protected:
  void AdamPreSolve();
  virtual void ComputeUpdateValue(int param_id, Dtype rate);

  DISABLE_COPY_AND_ASSIGN(AdamSolver);
};

}  // namespace caffe

#endif  // CAFFE_SGD_SOLVERS_HPP_
