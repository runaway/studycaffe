#include <boost/thread.hpp>
#include "caffe/layer.hpp"
/*
Layer���㣩��Caffe�����Ӵ���ӵ�ģ�顣����Caffeǿ��ģ�黯��ƣ����ֻ����ÿ��
layer���һ���ض��ļ��㣬����convolution������pooling�������Ա任���ڻ����㣬��
�����ݼ��ء���һ������ʧ����ȡ�layer��������˵���������յ�һ���������ˣ����
�����ؾ���һ��һ���layer���໥֮��ͨ��blob������������������
�����ȿ�һ��ͼ��

Ȼ�����Ǵ�ͷ�ļ�������
Caffe����Layer��ص�ͷ�ļ���7����
layer.hpp: ����Layer����������layer�Ļ����ӿڡ�
data_layers.hpp: �̳��Ը���Layer���������������ݲ�����ص���Layer������DataLayer��
HDF5DataLayer��ImageDataLayer�ȡ�
vision_layers.hpp: �̳��Ը���Layer�����������������ص���Layer������ConvolutionLayer��
PoolingLayer��LRNLayer�ȡ�
neuron_layers.hpp: �̳��Ը���Layer������������Ա任��ص���Layer������ReLULayer��
TanHLayer��SigmoidLayer�ȡ�
loss_layers.hpp: �̳��Ը���Layer�������������������ص���Layer������EuclideanLossLayer��
SoftmaxWithLossLayer��HingeLossLayer�ȡ�
common_layers.hpp: �̳��Ը���Layer���������м������ݱ��Ρ���Ԫ�ز�����ص���
Layer������ConcatLayer��InnerProductLayer��SoftmaxLayer�ȡ�

layer_factory.hpp: Layer����ģʽ�࣬����ά�����п���layer����Ӧlayer���췽����ӳ���
1.About
layer.hpp
��layer��ص�ͷ�ļ��У�
common_layers.hpp
data_layers.hpp
layer.hpp
loss_layers.hpp
neuron_layers.hpp
vision_layers.hpp
����``layer.hpp�ǳ�������Ļ��࣬����������������ϵļ̳У�Ҳ��ʣ�µ����ͷ�ļ�
����ͼ�е�������֡���layer.hpp`ͷ�ļ���������⼸��ͷ�ļ���
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/device_alternate.hpp"
��device_alternate.hpp�У�ͨ��#ifdef CPU_ONLY������һЩ����ȡ��GPU�ĵ��ã�
#define STUB_GPU(classname)
#define STUB_GPU_FORWARD(classname, funcname)
#define STUB_GPU_BACKWARD(classname, funcname)
layer������������Ҫ������
LayerParameter layer_param_;      // �����protobuf�ļ��д洢��layer����
vector<share_ptr<Blob<Dtype>>> blobs_;        // ����洢����layer�Ĳ������ڳ�
�����õ�
vector<bool> param_propagate_down_;        // ���bool��ʾ�Ƿ�������blob����
��diff�����������
Layer��Ĺ�������explicit Layer(const LayerParameter& param) : layer_param_(param)
�᳢�Դ�protobuf�ļ���ȡ��������������Ҫ�ӿڣ�
virtual void SetUp(const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top)
inline Dtype Forward(const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top);
inline void Backward(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const <Blob<Dtype>*>* bottom);
SetUp������Ҫ����ʵ�ʵĲ������ý���ʵ�֣��Ը������͵Ĳ�����ʼ����Forward��Backward��Ӧǰ�����ͷ�����£�����ͳһ����bottom�����Ϊtop������Backward�����и�propagate_down������������ʾ��Layer�Ƿ��򴫲�������
��Forward��Backward�ľ���ʵ��������Caffe::mode()���ж�Ӧ�Ĳ�������ʹ��cpu����gpu���м��㣬������ʵ���˶�Ӧ�Ľӿ�Forward_cpu��Forward_gpu��Backward_cpu��Backward_gpu����Щ�ӿڶ���virtual�����廹��Ҫ����layer�����ͽ��ж�Ӧ�ļ��㣨ע�⣺��Щlayer��û��GPU�����ʵ�֣����Է�װʱ������CPU�ļ�����Ϊ�󱸣������⣬��ʵ����ToProto�Ľӿڣ���Layer�Ĳ���д�뵽protocol buffer�ļ��С�
data_layers.hpp
data_layers.hpp���ͷ�ļ��������⼸��ͷ�ļ���
#include "boost/scoped_ptr.hpp"
#include "hdf5.h"
#include "leveldb/db.h"
#include "lmdb.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
����hdf5��leveldb��lmdb��ȷʵ���������������ˡ�data_layer��Ϊԭʼ���ݵ�����㣬���������������ײ㣬�����Դ����ݿ�leveldb��lmdb�ж�ȡ���ݣ�Ҳ����ֱ�Ӵ��ڴ��ж�ȡ�������Դ�hdf5��������ԭʼ��ͼ��������ݡ�
�����⼸�����ݿ⣬������£�
LevelDB��Google��˾���һ�������ܵ�key/value�洢�⣬���ü򵥣������Ǳ�Snappyѹ������˵Ч�ʺܶ࣬���Լ��ٴ���I/O���������ӿ��Կ���ά���ٿơ�
��LMDB��Lightning Memory-Mapped Database�����Ǹ���levelDB���Ƶ�key/value�洢�⣬��Ч���ƺ�����Щ������ҳ��д����ultra-fast��ultra-compact��������д���һ��ѧϰ������
HDF��Hierarchical Data Format����һ��Ϊ�洢�ʹ����������ѧ���ݶ���Ƶ��ļ���ʽ����Ӧ�Ŀ��ļ�����ǰ�����еİ汾��HDF5,���ļ��������ֻ������ݶ���
Ⱥ�飨group���������ļ��У����԰���������ݼ����¼�Ⱥ�飻
���ݼ���dataset�����������ݣ������Ƕ�ά���飬Ҳ�����Ǹ����ӵ��������͡�
������������ά���ٿƣ�����ʹ�ÿ��Բο�[HDF5 С�ԡ����ߴ��ϵĶ�����ļ���ʽ](HDF5 С�ԡ����ߴ��ϵĶ�����ļ���ʽ)������������ϸ���о�����ô�á�
caffe/filler.hpp���������������ʼ��ʱ������layer�Ķ�����г�ʼ��������䣬����Ĵ����ֱ�ۣ�����FillerParameterָ�������ͽ��ж�Ӧ�Ĳ�����䡣
// A function to get a specific filler from the specification given in
// FillerParameter. Ideally this would be replaced by a factory pattern,
// but we will leave it this way for now.
template <typename Dtype>
Filler<Dtype>* GetFiller(const FillerParameter& param) {
  const std::string& type = param.type();
  if (type == "constant") {
    return new ConstantFiller<Dtype>(param);
  } else if (type == "gaussian") {
    return new GaussianFiller<Dtype>(param);
  } else if (type == "positive_unitball") {
    return new PositiveUnitballFiller<Dtype>(param);
  } else if (type == "uniform") {
    return new UniformFiller<Dtype>(param);
  } else if (type == "xavier") {
    return new XavierFiller<Dtype>(param);
  } else {
    CHECK(false) << "Unknown filler name: " << param.type();
  }
  return (Filler<Dtype>*)(NULL);
}
internal_thread.hpp�����װ��pthread�������̳е�������Եõ�һ���������̣߳���Ҫ�������ڼ��㵱ǰ��һ������ʱ���ں�̨��ȡ��һ�������ݡ�
����data_layer������Ҫע����Ҷ���ͼƬ�ϱ�ע�ˡ�
neuron_layers.hpp
������data�󣬾�Ҫ�����ˣ����糣����sigmoid��tanh�ȵȣ���Щ������������������neuron_layers.hpp�������NeuronLayer�������ֻ�������ļ��㣬�����ȷ����������ExactNumBottomBlobs()��ExactNumTopBlobs()���ǳ���1,������һ��blob�����һ��blob��
common_layers.hpp
NeruonLayer��������򵥵�һ��һ���㣬��ʣ�µ���Щ���ӵļ�����ͨͨ������common_layers.hpp�С���ArgMaxLayer��ConcatLayer��FlattenLayer��SoftmaxLayer��SplitLayer��SliceLayer�ȸ��ֶ�blob�����޸ĵĲ�����
loss_layers.hpp
ǰ���data layer��common layer�����м����㣬��Ȼ���漰�����򴫲�����������Դͷ������loss_layer������������նˡ���һ����ΪҪ�������������붼��2��blob�����1��blob��
vision_layers.hpp
vision_layer��Ҫ��ͼ�����Ĳ�������convolusion��pooling��LRN�������棬���ٷ��ĵ���˵�����ǿ������ͼ��ģ����Ҫ������ʵ�ִ����ˡ������и�im2col��ʵ�֣���caffe���ߵĽ��ͣ���Ҫ��Ϊ�˼��پ���ġ�
layer_factory.hpp
layer_factory�Ƚ���Ҫ�Ҿͷ�����һƪ�����ˡ�
2. Detail
����һSection�У��������뵽��һС�������ļ���layer��ϸ����ȥ������һЩ���õ�layer�������㣬�ػ��㣨Pooling������������Ӧ��proto���롣
2.1. ���ݲ㣨data_layers��
����ͨ�����ݲ����Caffe�����ݲ�����������ĵײ������ݿ������Ը�Ч�����ݿ⣨LevelDB ���� LMDB����ֱ�������ڴ档�����׷���Ч�ԣ�������HDF5����һ��ͼ��ĸ�ʽ��Ӳ�̶�ȡ���ݡ�
һЩ�����Ĳ������磺mean subtraction, scaling, random cropping, and mirroring������ֱ�������ݲ��Ͻ���ָ����
1 Database
���ͣ�Data
���������
source: �������ݵ�Ŀ¼����
batch_size: һ�δ�������������
��ѡ������
rand_skip: �ڿ�ʼ��ʱ������������������ֵ�������첽����ݶ��½���SGD����ʱ��ǳ�����
backend [default LEVELDB]: ѡ��ʹ�� LEVELDB ���� LMDB
2 In-Memory
����: MemoryData
���������
batch_size, channels, height, width: ָ�����ڴ��ȡ���ݵĴ�С
MemoryData��ֱ�Ӵ��ڴ��ж�ȡ���ݣ������ǿ�����������ˣ�Ҫʹ�����Ļ�����������MemoryDataLayer::Reset (from C++)����Net.set_input_arrays (from Python)�Դ�ָ��һ�����������ݣ�ͨ����һ����ά��������
3 HDF5 Input
����: HDF5Data
��Ҫ������
source: ��Ҫ��ȡ���ļ���
batch_size��һ�δ�������������
4 HDF5 Output
����: HDF5Output
��Ҫ������
file_name: ������ļ���
HDF5�����ú�����е������Ĳ㲻һ�������ǰ������blobsд��Ӳ��
5 Images
����: ImageData
��Ҫ������
source: text�ļ������֣�ÿһ�и���һ��ͼƬ���ļ�����label
batch_size: һ��batch��ͼƬ������
��ѡ������
rand_skip���ڿ�ʼ��ʱ������������������ֵ�������첽����ݶ��½���SGD����ʱ��ǳ�����
shuffle [default false]
new_height, new_width: �����е�ͼ��resize�������С
6 Windows
���ͣ�WindowData
7 Dummy
���ͣ�DummyData
Dummy ������development ��debugging���������DummyDataParameter��
2.2. �����㣨neuron_layers��
һ����˵����������element-wise�Ĳ��������������Ĵ�С��ͬ��һ������¾���һ�������Ժ�����
���룺
n��c��h��w
�����
n��c��h��w
1 ReLU / Rectified-Linear and Leaky-ReLU
����: ReLU
����:
<span style="font-family:Microsoft YaHei;font-size:12px;">layer {
  name: "relu1"
  type: "ReLU"
  bottom: "conv1"
  top: "conv1"
}</span>
��ѡ������
negative_slope [default 0]�� ָ������ֵС����ʱ�������
ReLU��Ŀǰʹ������ļ�����������Ҫ��Ϊ���������죬�����ܱ���ͬ��Ч������׼��ReLU����Ϊmax(x, 0)����һ��Ϊ��x > 0ʱ���x����x <= 0ʱ���negative_slope��RELU��֧��in-place���㣬����ζ��bottom�������������ͬ�Ա����ڴ�����ġ�
ReLU(x)=max{0,x}
ReLU Function
2 Sigmoid
���ͣ�Sigmoid
���ӣ�
<span style="font-family:Microsoft YaHei;font-size:12px;">layer {
  name: "encode1neuron"
  bottom: "encode1"
  top: "encode1neuron"
  type: "Sigmoid"
}</span>
Sigmoid��ͨ�� sigmoid(x) ����ÿһ������x���������������ͼ��
��(x)=11+exp?x
����дͼƬ����
3 TanH / Hyperbolic Tangent
����: TanH
����:
<span style="font-family:Microsoft YaHei;font-size:12px;">layer {
  name: "layer"
  bottom: "in"
  top: "out"
  type: "TanH"
}</span>
TanH��ͨ�� tanh(x) ����ÿһ������x���������������ͼ����ע��sigmoid������TanH�����������ϵ�����sigmoid������ʵ��ӳ�䵽(0,1)��TanH��ʵ��ӳ�䵽(-1,1)��
tanh(x)=expx?exp?xexpx+exp?x
����дͼƬ����
4 Absolute Value
����: AbsVal
����:
<span style="font-family:Microsoft YaHei;font-size:12px;">layer {
  name: "layer"
  bottom: "in"
  top: "out"
  type: "AbsVal"
}</span>
ABSVAL��ͨ�� abs(x) ����ÿһ������x�������
5 Power
���ͣ� Power
���ӣ�
<span style="font-family:Microsoft YaHei;font-size:12px;">layer {
  name: "layer"
  bottom: "in"
  top: "out"
  type: "Power"
  power_param {
    power: 1
    scale: 1
    shift: 0
  }
}</span>
��ѡ������
power [default 1]
scale [default 1]
shift [default 0]
POWER��ͨ�� (shift + scale * x) ^ power����ÿһ������x�������
6 BNLL
����: BNLL
���ӣ�
<span style="font-family:Microsoft YaHei;font-size:12px;">layer {
  name: "layer"
  bottom: "in"
  top: "out"
  type: BNLL
}</span>
BNLL (binomial normal log likelihood) ��ͨ�� log(1 + exp(x)) ����ÿһ������x�������
2.3. �Ӿ��㣨vision_layers��
1 �����(Convolution)
���ͣ�Convolution
���ӣ�
<span style="font-family:Microsoft YaHei;font-size:12px;">layers { 
    name: "conv1" 
    type: CONVOLUTION 
    bottom: "data" 
    top: "conv1" 
    blobs_lr: 1               # learning rate multiplier for the filters 
    blobs_lr: 2               # learning rate multiplier for the biases 
    weight_decay: 1           # weight decay multiplier for the filters 
    weight_decay: 0           # weight decay multiplier for the biases 
    convolution_param { 
        num_output: 96        # learn 96 filters 
        kernel_size: 11       # each filter is 11x11 
        stride: 4             # step 4 pixels between each filter application 
        weight_filler { 
            type: "gaussian"  # initialize the filters from a Gaussian 
            std: 0.01         # distribution with stdev 0.01 (default mean: 0) } 
            bias_filler { 
                type: "constant" # initialize the biases to zero (0) 
                value: 0 
            } 
        }
    }
}</span>
blobs_lr: ѧϰ�ʵ����Ĳ����������������������Ȩ��ѧϰ�ʺ������������������ѧϰ��һ����ͬʱ��ƫ��ѧϰ��ΪȨ�ص�������
weight_decay��
��������Ҫ����
���������
num_output (c_o)���������ĸ���
kernel_size (or kernel_h and kernel_w)���������Ĵ�С��Ҳ������ν���ˡ��Ĵ�С����
���������
weight_filler [default type: ��constant�� value: 0]�������ĳ�ʼ������
��ѡ������
bias_filler��ƫ�õĳ�ʼ������
bias_term [default true]��ָ���Ƿ��Ƿ���ƫ����
pad (or pad_h and pad_w) [default 0]��ָ���������ÿһ�߼��϶��ٸ�����
stride (or stride_h and stride_w) [default 1]��ָ���������Ĳ���
group (g) [default 1]: ���g>1����ô��ÿ���˲������޶�ֻ��ĳ��������Ӽ��й��������仰˵���������Ϊg�飬ͬʱ�����Ҳ��Ϊg�顣��ô��i�����ֻ���i�������йء�
ͨ�������Ĵ�С�仯��
���룺
n��ci��hi��wi
�����
n��co��ho��wo
���У�ho=(hi+2��padh?kernelh)/strideh+1��woͨ��ͬ���ķ������㡣
2 �ػ��㣨Pooling��
���ͣ�Pooling
���ӣ�
<span style="font-family:Microsoft YaHei;font-size:12px;">layers { 
    name: "pool1" 
    type: POOLING 
    bottom: "conv1" 
    top: "pool1" 
    pooling_param { 
        pool: MAX 
        kernel_size: 3 # pool over a 3x3 region 
        stride: 2 # step two pixels (in the bottom blob) between pooling regions 
    }
}</span>
��������Ҫ����
���������
kernel_size (or kernel_h and kernel_w)���������Ĵ�С
��ѡ������
pool [default MAX]��pooling�ķ�����Ŀǰ��MAX, AVE, ��STOCHASTIC���ַ���
pad (or pad_h and pad_w) [default 0]��ָ���������ÿһ����϶��ٸ�����
stride (or stride_h and stride_w) [default 1]��ָ���������Ĳ���
ͨ���ػ���Ĵ�С�仯��
���룺
n��ci��hi��wi
�����
n��co��ho��wo
���У�ho=(hi+2��padh?kernelh)/strideh+1��woͨ��ͬ���ķ������㡣
3 Local Response Normalization (LRN)
���ͣ�LRN
��ѡ������
local_size [default 5]������cross channel LRNΪ��Ҫ��͵��ڽ�channel������������within channel LRNΪ��Ҫ��͵Ŀռ�����ı߳���
alpha [default 1]��scaling������
beta [default 5]��ָ����
norm_region [default ACROSS_CHANNELS]: ѡ��LRNʵ�ֵķ�����1. ACROSS_CHANNELS ��2. WITHIN_CHANNEL
LRN��Local Response Normalization���Ƕ�һ���ֲ�������������еĹ�һ���������ֲ�ͬ����ʽ��1. ACCROSS_CHANNEL��2. WITHIN_CHANNEL����ʵ�ܺô������Ͻ�����⡣��һ�ַ����ۺ��˲�ͬ��channel������һ��channel����ֻȡ1*1������size��localsize��1��1�������ڵڶ��ַ����У�����channel��������չ��ֻ�ڵ�һchannel�Ͻ��пռ���չ������size��1��localsize��localsize����
���㹫ʽ����ÿһ���������(1+(��/n)?��ix2i)��
�������������scaling��������������ָ����������n��Ӧlocal region�Ĵ�С��
2.4. ��ʧ�㣨Loss Layers��
���ѧϰ��ͨ����С�������Ŀ���Loss������ѧϰ��
1 Softmax
����: SoftmaxWithLoss
����Softmax�����ݣ����Բο���֮ǰ�Ĳ��ͣ�������ѧϰ��Softmax Regression��顣Softmax Loss��Ӧ���ڶ��ǩ���ࡣ�������룬������multinomial logistic loss���ڸ����Ͻ��Ƶ���һ��Softmax�����һ��multinomial logistic loss�㡣�����ݶȵļ����ϸ����ȶ���
2 Sum-of-Squares / Euclidean
����: EuclideanLoss
Euclidean loss�����������������ƽ���ͣ�
12N��i=1N||x1i?x2i||2x
3 Hinge / Margin
����: HingeLoss
���ӣ�
<span style="font-family:Microsoft YaHei;font-size:12px;">L1 Normlayers { 
    name: "loss" 
    type: HINGE_LOSS 
    bottom: "pred" 
    bottom: "label"
} 
L2 Normlayers { 
    name: "loss" 
    type: HINGE_LOSS 
    bottom: "pred" 
    bottom: "label" 
    top: "loss" 
    hinge_loss_param { 
        norm: L2 
    }
}</span>
��ѡ������
norm [default L1]: ѡ��L1����L2����
���룺
n��c��h��w Predictions
n��1��1��1 Labels
���
1��1��1��1 Computed Loss
4 Sigmoid Cross-Entropy
���ͣ�SigmoidCrossEntropyLoss
5 Infogain
���ͣ�InfoGainLoss
6 Accuracy and Top-k
���ͣ�Accuracy
�������������Ŀ�����ȷ�ʣ���ʵ���ⲻ��һ��loss������û��backward��һ����
2.5. һ��㣨Common Layers��
1 ȫ���Ӳ� Inner Product
���ͣ�InnerProduct
���ӣ�
<span style="font-family:Microsoft YaHei;font-size:12px;">layer {
  name: "fc8"
  type: "InnerProduct"
  # learning rate and decay multipliers for the weights
  param { lr_mult: 1 decay_mult: 1 }
  # learning rate and decay multipliers for the biases
  param { lr_mult: 2 decay_mult: 0 }
  inner_product_param {
    num_output: 1000
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
  bottom: "fc7"
  top: "fc8"
}</span>
��Ҫ������
num_output (c_o)���������ĸ���
��ѡ������
weight_filler [default type: ��constant�� value: 0]�������ĳ�ʼ������
bias_filler��ƫ�õĳ�ʼ������
bias_term [default true]��ָ���Ƿ��Ƿ���ƫ����
ͨ��ȫ���Ӳ��Ĵ�С�仯��
���룺n��ci��hi��wi
�����n��co��1��1
2 Splitting
���ͣ�Split
Splitting����԰�һ������blob����ɶ�����blobs��������ڵ���Ҫ��һ��blob���뵽���������ʱ��
3 Flattening
���ͣ�Flatten
Flatten���ǰ�һ������Ĵ�СΪn * c * h * w���һ���򵥵����������СΪ n * (c*h*w) * 1 * 1��
4 Reshape
���ͣ�Reshape
���ӣ�
<span style="font-family:Microsoft YaHei;font-size:12px;">  layer {
    name: "reshape"
    type: "Reshape"
    bottom: "input"
    top: "output"
    reshape_param {
      shape {
        dim: 0  # copy the dimension from below
        dim: 2
        dim: 3
        dim: -1 # infer it from the other dimensions
      }
    }
  }</span>
���룺������һ��blob������������ά��
�����ͬ����blob����������ά���Ѿ���������Ϊ�ظı䣬ά�ȵ�������reshap_param���塣
��ѡ������
shape
Reshape�㱻���ڸı������ά�ȣ������ı�����ľ������ݡ�����Flatten��һ����ֻ��ά�ȱ��ı���ѣ�������̲��漰���ݵĿ�����
�����ά����ReshapeParam proto���ơ�����ֱ��ʹ�����ֽ���ָ�����趨�����ĳһά�����blob��ȥ�����⣬������������ֵ��˵һ�£�
0 ֱ�Ӵӵײ㸴�ơ����磬����ǵײ���һ��2�����ĵ�һά����ô���������ĵ�һάҲ��һ��2��
-1 �����������������Ʋ���һάӦ���Ƕ��١�
5 Concatenation
���ͣ�Concat
���ӣ�
<span style="font-family:Microsoft YaHei;font-size:12px;">layer {
  name: "concat"
  bottom: "in1"
  bottom: "in2"
  top: "out"
  type: "Concat"
  concat_param {
    axis: 1
  }
}</span>
��ѡ������
axis [default 1]��0��������num��1��������channels
ͨ��ȫ���Ӳ��Ĵ�С�仯��
���룺��1��K��ÿһ��blob�Ĵ�С��ni��ci��h��w
�����
���axis = 0: (n1+n2+...+nK)��c1��h��w����Ҫ��֤���������ci��ͬ��
���axis = 1: n1��(c1+c2+...+cK)��h��w����Ҫ��֤���������n_i ��ͬ��
ͨ��Concatenation�㣬���԰Ѷ����blobs���ӳ�һ��blob��
6 Slicing
���ͣ�Slice
���ӣ�
<span style="font-family:Microsoft YaHei;font-size:12px;">layer {
  name: "slicer_label"
  type: "Slice"
  bottom: "label"
  ## Example of label with a shape N x 3 x 1 x 1
  top: "label1"
  top: "label2"
  top: "label3"
  slice_param {
    axis: 1
    slice_point: 1
    slice_point: 2
  }
}</span>
Slice����Խ�������ɶ������㡣��Щ�������һ��������ά�ȴ��ڡ�axisָ����Ŀ����ᣬslice_point��ָ����ѡ��ά�ȵ���š�
7 Elementwise Operations
���ͣ�Eltwise
8 Argmax
���ͣ�ArgMax
9 Softmax
���ͣ�Softmax
10 Mean-Variance Normalization
���ͣ�MVN

ÿ��layer��������������һЩ'bottom' blobs, ���һЩ'top' blobs��Caffe��ÿ������layer�Ĳ���˵��������caffe.proto�ļ��У������layer����ֵ�����ھ���Ӧ�õ�protocals buffer����ṹ˵���ļ��С����磬����㣨ConvolutionLayer���Ĳ���˵����caffe.proto�������¶���ģ�

���еĲ���˵����������˵ĸ�������С�Ͳ����ȡ���examples\mnist\lenet_train_test.prototxt����ṹ˵���ļ��У�����һ������㣨ConvolutionLayer������������ģ�

[cpp] view plain copy ��CODE�ϲ鿴����Ƭ�������ҵĴ���Ƭ
# in examples\mnist\lenet_train_test.prototxt  
layer {  
  name: "conv1" // �������  
  type: "Convolution" // ������ͣ�˵������ִ����һ�ּ���  
  bottom: "data" // �����������Blob������  
  top: "conv1" // ����������Blob������  
  param { // ���Ȩֵ��ƫ����ز���  
    lr_mult: 1  
  }  
  param {  
    lr_mult: 2  
  }  
  convolution_param { // �������������صĲ���  
    num_output: 20  
    kernel_size: 5  
    stride: 1  
    weight_filler {  
      type: "xavier"  
    }  
    bias_filler {  
      type: "constant"  
    }  
  }  
}  

ÿ�����͵�layer��Ҫ�������ֹؼ�����LayerSetUp, Forward, Backward��
LayerSetUp: ���繹��ʱ��ʼ����Ͳ������
Forward: ��������ǰ�򴫵ݣ�����bottom�������ݣ����������top
Backward�� �������򴫵ݣ�����top���ݶȣ�����bottom���ݶȲ��洢��bottom blob
Layer�������Ҫ����SetUp��Forward��Backward��������һ��ʼ��ʱ������á�Ȼ�����ǰ���ͷ�����
�����е�SetUp��ʵ����������CheckBlobCounts��LayerSetUp��Reshape�ȵ�ʵ�֡�������Reshape���Ǳ���Ҫʵ�ֵģ���Ϊ���Ǵ��麯��
�����е�Forward����������Forward_cpu��Forward_gpu��������Forward_cpu���Ǳ���Ҫʵ�ֵġ�
�����е�Backward����������Backward_cpu��Backward_gpu��������Backward_cpu ���Ǳ���Ҫʵ�ֵġ�
=================================================================================================================================
����layer����Ҫʵ��һ��forward function��ǰ�ݺ�����Ȼ���ܿ����Լ�����������forward���������inputҲ����Layer��bottom������caffe���������ǰһ���ǽ�bottom�ģ���bottom�л�ȡblob�����Ҽ��������Blob����Ȼ����Ҳ��ʵ��һ�����򴫲����������ǵ�input��blob�Լ�output blob��error gradient �ݶ�������õ��ò���ݶ����ӹ�ʽ��Ҳ���Կ�����
��l=((wl+1)T��l+1)�ҡ�(zl)
��ѧ��caffe���鿴Դ�룬layer.hpp:

*/

namespace caffe {

// template <typename Dtype>
// ��ʼ��������  
template <typename Dtype>  
void Layer<Dtype>::InitMutex() {  
  forward_mutex_.reset(new boost::mutex());  
}  
  
// Lock  
template <typename Dtype>  
void Layer<Dtype>::Lock() {  
  if (IsShared()) {  
    forward_mutex_->lock();  
  }  
}  
  
// UnLock  
template <typename Dtype>  
void Layer<Dtype>::Unlock() {  
  if (IsShared()) {  
    forward_mutex_->unlock();  
  }  
}  

INSTANTIATE_CLASS(Layer);

}  // namespace caffe
