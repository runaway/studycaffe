#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

/*
Common Layers：一般层
 

Inner Product
layer类型：InnerProduct
CPU实现：./src/caffe/layers/inner_product_layer.cpp
CUDA GPU实现：./src/caffe/layers/inner_product_layer.cu
参数(InnerProductParameter inner_product_param)
必需的
num_output (c_o): 滤波器数目
强烈推荐的
weight_filler [default type: 'constant' value: 0]
可选的
bias_filler [default type: 'constant' value: 0]
bias_term [default true]: 指定是否对滤波器输出学习和应用一组附加偏差项
输入：n * c_i * h_i * w_i
输出：n * c_o * 1 * 1
例子
复制代码
layer {
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
}

复制代码
内积层（实际上通常指全连接层）将输入看成简单向量，产生一个单个向量形式的输出（blob的高和宽设置为1）。

Splitting：分割
分割层是一个功能层，将输入blob分成多个输出blob。这个layer用于一个blob被输入到多个输出层的情况。

Flattening：压扁
flatten layer也是一个功能层，将形为n * c * h * w的blob输入压扁成一个形为n * (c * h * w)的简单向量，
实际上是单独压缩，每个数据是一个简单向量，维度c * h * w，共n个向量。

Reshape：整形
layer类型：Reshape
CPU实现：./src/caffe/layers/reshape_layer.cpp
参数(ReshapeParameter reshape_param)
可选的
shape
输入：一个任意维度的blob
输出：同一个blob，维度修改为reshape_param
例子：
复制代码
  layer {
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
  }
复制代码
reshape layer用于改变输入维度，但是不改变数据。就像flatten layer一样，仅仅数据维
度改变，过程中没有数据被拷贝。

输出维度被Reshape_param指定。帧数直接使用，设置相应的输出blob的维度。在目标维度
值设置时，两个特殊值被接受：
0： 从bottom layer拷贝相应维度。如果给定dim: 0，且bottom由2作为第一维维度，那么
top layer也由2作为第一维维度 ==> 不改变原始维度
-1：代表从其他维度推断这一维维度。这个行为与numpy的-1和Matlab reshape时的[ ]作用
是相似的。维度被计算，使得总体输出维度与bottom layer相似。在reshape操作中至多可
以设置一个-1。
另外一个例子，指定reshape_param{shape{dim: 0 dim:-1}}作用与Flatten layer作用相同，
都是将输入blob压扁成向量。

Concatenation：拼接

concat layer是一个功能层，用于将多个输入blob拼接城一个单个的输出blob。

layer类型：Concat
CPU实现：./src/caffe/layers/concat_layer.cpp
CUDA GPU实现：./src/caffe/layers/concat_layer.cu
参数(ConcatParameter concat_param)
可选的
axis [default 1]: 0表示沿着num连接，1表示按通道连接。
输入：n_i * c_i * h * w，K个输入blob
输出：
如果axis = 0: (n_1 + n_2 + ... + n_K) * c_1 * h * w，所有输入的c_i应该相同；
如果axis = 1: n_1 * (c_1 + c_2 + ... + c_K) * h * w，所有输入的n_i应该相同。
例子：
复制代码
layer {
  name: "concat"
  bottom: "in1"
  bottom: "in2"
  top: "out"
  type: "Concat"
  concat_param {
    axis: 1
  }
}
复制代码
 

Slicing：切片
slice layer也是一个功能层，将一个输入层沿着给定维度（当前仅提供基于num和通道的实现）切片成多个输出层。

例子：

复制代码
layer {
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
}
复制代码
axis表示目标axis，沿着给定维度切片。slice_point表示选择维度的索引，索引数目应该等于顶层blob数目减一。

Elementwise Operations
Eltwise

Argmax
ArgMax

Softmax
Softmax

Mean-Variance Normalization
MVN

*/


namespace caffe {

template <typename Dtype>
void InnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
    const int num_output = this->layer_param_.inner_product_param().num_output();
    bias_term_ = this->layer_param_.inner_product_param().bias_term();
    transpose_ = this->layer_param_.inner_product_param().transpose();

    // 全连接层输出神经元的个数  
    N_ = num_output;
    const int axis = bottom[0]->CanonicalAxisIndex(
    this->layer_param_.inner_product_param().axis());

    // K_ 单个样本特征向量长度  
    // Dimensions starting from "axis" are "flattened" into a single
    // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
    // and axis == 1, N inner products with dimension CHW are performed.
    K_ = bottom[0]->count(axis);

    // 在Caffe.proto里，LayerParameter中有一个repeated blobs field，但是在更多
    // net的定义文件即prototxt文件里并没有blobs，那么在这里将进行处理__显然，如
    // 果this->blobs_.size()>0那么参数blob就不需要初始化了,skip;反之,则进行初始化  
    // Check if we need to set up the weights
    if (this->blobs_.size() > 0) 
    {
        LOG(INFO) << "Skipping parameter initialization";
    } 
    else 
    {
        if (bias_term_) 
        {
            this->blobs_.resize(2);
        } 
        else 
        {
            this->blobs_.resize(1);
        }
    
        // Initialize the weights
        vector<int> weight_shape(2);

        if (transpose_) 
        {
            weight_shape[0] = K_;
            weight_shape[1] = N_;
        } 
        else 
        {
            weight_shape[0] = N_;
            weight_shape[1] = K_;
        }
    
        // 可以认为blobs_[0]的维度为N_*K_，即通常，我们将权值矩阵设为N*K维。可以
        // 这么认为，但是在实际上，在C++中数据都是存放在内存中，并没有所谓的矩阵的概念
        this->blobs_[0].reset(new Blob<Dtype>(weight_shape));

        // fill the weights 定义了一个智能指针
        shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_param().weight_filler()));

        weight_filler->Fill(this->blobs_[0].get());

        // If necessary, intiialize and fill the bias term
        if (bias_term_) 
        {
            vector<int> bias_shape(1, N_);
            this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
            
            shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
              this->layer_param_.inner_product_param().bias_filler()));

            bias_filler->Fill(this->blobs_[1].get());
        }
    }  // parameter initialization

    // param_propagate_down_是从Layer<Dtype> 继承来的数据成员  
    this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
    // Figure out the dimensions
    const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
    const int new_K = bottom[0]->count(axis);
    CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";

    // 若axis=1,则M_表示bottom[0]里的样本个数  
    // The first "axis" dimensions are independent inner products; the total
    // number of these is M_, the product over these dimensions.
    M_ = bottom[0]->count(0, axis);
    
    // The top shape will be the bottom shape with the flattened axes dropped,
    // and replaced by a single axis with dimension num_output (N_).
    vector<int> top_shape = bottom[0]->shape();

    // 重置top_shape，对于全链接层输出top往往不需要像bottom那样四维（NxCxHxW），
    // 所以重置。如果axis=1,那么top就重置为二维的，即一个矩阵。注意vector的
    // resize操作__此种情况下，axis之前的元素保持不变  
    top_shape.resize(axis + 1);

    // 为矩阵的第二维赋值，即矩阵的列数;矩阵的行数为M_  
    top_shape[axis] = N_;

    // top[0]的shape变成了M_x N_  
    top[0]->Reshape(top_shape);

    // Set up the bias multiplier
    if (bias_term_) 
    {
        vector<int> bias_shape(1, M_);
        bias_multiplier_.Reshape(bias_shape);
        caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
    }
}

/*
forward实现的功能就是 y=xw'+b  
x为输入，维度 MxK  
y为输出，维度 Nx1  
w为权重，维度 NxK  
b为偏置，维度 Nx1  
具体到代码实现，用的是这个函数caffe_cpu_gemm，具体的函数头为:  
void caffe_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,  
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,  
    const float alpha, const float* A, const float* B, const float beta,  
    float* C)  
整理它的功能其实很直观，即C←α* op(A)×op(B)+β*C  
const CBLAS_TRANSPOSE TransA  # A是否转置  
const CBLAS_TRANSPOSE TransB  # B是否转置  
若TransA = CblasNoTrans, op( A ) = A；若TransA = CblasTrans, op( A ) = A'  
M N K个人觉得为：  
const int M <strong>//op()操作后矩阵A的行数，矩阵C的行数 op()操作一般为转置或共轭转置  
const int N <strong>//op()操作后矩阵B的列数，矩阵C的列数 
const int K <strong>//op()操作后矩阵A的列数，矩阵B的行数
则，其中A维度是MxK，B维度是KxN，C维度为MxN  
lda，ldb，ldc，在BLAS的文档里，这三个参数分别为ABC的行数，但是实际使用发现，在
CBLAS里应该是列数，注意是经过op()操作的矩阵ABC的列数  

全连接层的forward包括了两步:  
# 这一步表示 y←wx，或者说是y←xw'  
caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,  
      bottom_data, weight, (Dtype)0., top_data);  
# 这一步表示 y←y+b  
caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,  
        bias_multiplier_.cpu_data(),  
        this->blobs_[1]->cpu_data(), (Dtype)1., top_data);<pre code_snippet_id="1584186" snippet_file_name="blog_20160221_4_1082892" name="code" class="cpp">实际上参与的计算为：(Mx1) x (1xN) = MxN   在C++中，数据都是存储在内存中，并以指针指向，那么为什么一个是<span style="font-family: Arial, Helvetica, sans-serif;">Mx1，一个是1xN, 个人觉得这应该是处于向量相对于矩阵的特殊性，更加自由点——以一维数组和多维数组为例，一维数组的数据在内存中的存储形式比多维数组的要自由点，约束少点，因为多维数组要保持RowMajor</span>  
*/
template <typename Dtype>
void InnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) 
{
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    const Dtype* weight = this->blobs_[0]->cpu_data();
    
    caffe_cpu_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
      M_, N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., top_data);
    
    if (bias_term_) 
    {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
            bias_multiplier_.cpu_data(),
            this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
    }
}

/*
参考UFLDL上的公式  
第一步，更新w，对应代码是：  
caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,  
        top_diff, bottom_data, (Dtype)0., this->blobs_[0]->mutable_cpu_diff());  
        
对照公式，有：  
需要更新的w的梯度的维度是NxK  
公式中的a^(l)对应的是bottom_data，维度是MxK  
公式中的\delta_(l+1)对应的是top_diff，维度是MxN  
  
第二步，更新b，对应代码是：  
caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,  
        bias_multiplier_.cpu_data(), (Dtype)0.,  
        this->blobs_[1]->mutable_cpu_diff());  
对照公式，有：  
公式中，b的梯度的维度应该为Nx1 ; \delta_(l+1)对应的是top_diff，维度是MxN  
这里用到了caffe_cpu_gemv，简单来说跟上面的caffe_cpu_gemm类似，不过前者是计算矩阵
和向量之间的乘法的（从英文命名可以分辨，v for vector, m for matrix）。函数头：  
void caffe_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,  
    const int N, const float alpha, const float* A, const float* x,  
    const float beta, float* y)   
# <strong>实现的功能类似 Y←αAX + βY，若需要转置，则Y←αA'X + βY.所以个人认
为ablas_sgemv()中参数MN表示的在op()操作之前的时候矩阵的行数和列数，即不管是
Y←αAX + βY还是Y←αA'X + βY，都是矩阵A本身的行数和列数，而非其转置。  
# 其中A的维度为 MxN  
# X是一个向量，维度为 Mx1  
# Y是结果 ，也是一个向量，维度为Nx1</strong>  
const CBLAS_TRANSPOSE TransA  # 是否对A进行转置  
# 下面的参数很直观，不描述了  
const int M  
const int N  
const float alpha  
const float* A  
const float* x  
const float beta  
float* y  
绕回到具体的代码实现。。如何更新b？根据公式b的梯度直接就是delta  
# 所以对应的代码其实就是将top_diff转置后就可以了（忽略乘上bias_multiplier这步）  
caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,  
        bias_multiplier_.cpu_data(), (Dtype)0.,  
        this->blobs_[1]->mutable_cpu_diff());  
进行的计算实际为：(MxN)' x (Mx1) = N x 1  
  
第三步是计算\delta^(l)：  
<strong>在公式中有一项f’，这里面可以忽略掉最后一项f’。因为在caffe实现中，这是
由Relu layer来实现的，这里只需要实现括号里面的累加就好了，这个累加其实可以等价
于矩阵乘法：  
caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,  
        top_diff, this->blobs_[0]->cpu_data(), (Dtype)0.,  
        (*bottom)[0]->mutable_cpu_diff());  
# top_diff为\delta^(l+1) 维度 MxN  
# this->blobs_[0]->cpu_data()为W^(l) 维度 NxK  
# (*bottom)[0]->mutable_cpu_diff()是要计算的结果，也就是\delta^(l) 维度是MxK  
#即，当前层的\delta^(l) 维度是MxK，下一层的\delta^(l+1) 维度是MxN  
 
*/
template <typename Dtype>
void InnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) 
{
    if (this->param_propagate_down_[0]) 
    {
        const Dtype* top_diff = top[0]->cpu_diff();
        const Dtype* bottom_data = bottom[0]->cpu_data();
        
        // Gradient with respect to weight
        if (transpose_) 
        {
            caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
            K_, N_, M_,
            (Dtype)1., bottom_data, top_diff,
            (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
        } 
        else 
        {
            caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
            N_, K_, M_,
            (Dtype)1., top_diff, bottom_data,
            (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
        }
    }

    if (bias_term_ && this->param_propagate_down_[1]) 
    {
        const Dtype* top_diff = top[0]->cpu_diff();

        // Gradient with respect to bias
        caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.cpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_cpu_diff());
    }

    if (propagate_down[0]) 
    {
        const Dtype* top_diff = top[0]->cpu_diff();

        // Gradient with respect to bottom data
        if (transpose_) 
        {
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
            M_, K_, N_,
            (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
            (Dtype)0., bottom[0]->mutable_cpu_diff());
        } 
        else 
        {
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
            M_, K_, N_,
            (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
            (Dtype)0., bottom[0]->mutable_cpu_diff());
        }
    }
}

#ifdef CPU_ONLY
STUB_GPU(InnerProductLayer);
#endif

INSTANTIATE_CLASS(InnerProductLayer);
REGISTER_LAYER_CLASS(InnerProduct);

}  // namespace caffe
