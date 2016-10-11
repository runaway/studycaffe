#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

/*
Common Layers��һ���
 

Inner Product
layer���ͣ�InnerProduct
CPUʵ�֣�./src/caffe/layers/inner_product_layer.cpp
CUDA GPUʵ�֣�./src/caffe/layers/inner_product_layer.cu
����(InnerProductParameter inner_product_param)
�����
num_output (c_o): �˲�����Ŀ
ǿ���Ƽ���
weight_filler [default type: 'constant' value: 0]
��ѡ��
bias_filler [default type: 'constant' value: 0]
bias_term [default true]: ָ���Ƿ���˲������ѧϰ��Ӧ��һ�鸽��ƫ����
���룺n * c_i * h_i * w_i
�����n * c_o * 1 * 1
����
���ƴ���
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

���ƴ���
�ڻ��㣨ʵ����ͨ��ָȫ���Ӳ㣩�����뿴�ɼ�����������һ������������ʽ�������blob�ĸߺͿ�����Ϊ1����

Splitting���ָ�
�ָ����һ�����ܲ㣬������blob�ֳɶ�����blob�����layer����һ��blob�����뵽��������������

Flattening��ѹ��
flatten layerҲ��һ�����ܲ㣬����Ϊn * c * h * w��blob����ѹ���һ����Ϊn * (c * h * w)�ļ�������
ʵ�����ǵ���ѹ����ÿ��������һ����������ά��c * h * w����n��������

Reshape������
layer���ͣ�Reshape
CPUʵ�֣�./src/caffe/layers/reshape_layer.cpp
����(ReshapeParameter reshape_param)
��ѡ��
shape
���룺һ������ά�ȵ�blob
�����ͬһ��blob��ά���޸�Ϊreshape_param
���ӣ�
���ƴ���
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
���ƴ���
reshape layer���ڸı�����ά�ȣ����ǲ��ı����ݡ�����flatten layerһ������������ά
�ȸı䣬������û�����ݱ�������

���ά�ȱ�Reshape_paramָ����֡��ֱ��ʹ�ã�������Ӧ�����blob��ά�ȡ���Ŀ��ά��
ֵ����ʱ����������ֵ�����ܣ�
0�� ��bottom layer������Ӧά�ȡ��������dim: 0����bottom��2��Ϊ��һάά�ȣ���ô
top layerҲ��2��Ϊ��һάά�� ==> ���ı�ԭʼά��
-1�����������ά���ƶ���һάά�ȡ������Ϊ��numpy��-1��Matlab reshapeʱ��[ ]����
�����Ƶġ�ά�ȱ����㣬ʹ���������ά����bottom layer���ơ���reshape�����������
������һ��-1��
����һ�����ӣ�ָ��reshape_param{shape{dim: 0 dim:-1}}������Flatten layer������ͬ��
���ǽ�����blobѹ���������

Concatenation��ƴ��

concat layer��һ�����ܲ㣬���ڽ��������blobƴ�ӳ�һ�����������blob��

layer���ͣ�Concat
CPUʵ�֣�./src/caffe/layers/concat_layer.cpp
CUDA GPUʵ�֣�./src/caffe/layers/concat_layer.cu
����(ConcatParameter concat_param)
��ѡ��
axis [default 1]: 0��ʾ����num���ӣ�1��ʾ��ͨ�����ӡ�
���룺n_i * c_i * h * w��K������blob
�����
���axis = 0: (n_1 + n_2 + ... + n_K) * c_1 * h * w�����������c_iӦ����ͬ��
���axis = 1: n_1 * (c_1 + c_2 + ... + c_K) * h * w�����������n_iӦ����ͬ��
���ӣ�
���ƴ���
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
���ƴ���
 

Slicing����Ƭ
slice layerҲ��һ�����ܲ㣬��һ����������Ÿ���ά�ȣ���ǰ���ṩ����num��ͨ����ʵ�֣���Ƭ�ɶ������㡣

���ӣ�

���ƴ���
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
���ƴ���
axis��ʾĿ��axis�����Ÿ���ά����Ƭ��slice_point��ʾѡ��ά�ȵ�������������ĿӦ�õ��ڶ���blob��Ŀ��һ��

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

    // ȫ���Ӳ������Ԫ�ĸ���  
    N_ = num_output;
    const int axis = bottom[0]->CanonicalAxisIndex(
    this->layer_param_.inner_product_param().axis());

    // K_ ��������������������  
    // Dimensions starting from "axis" are "flattened" into a single
    // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
    // and axis == 1, N inner products with dimension CHW are performed.
    K_ = bottom[0]->count(axis);

    // ��Caffe.proto�LayerParameter����һ��repeated blobs field�������ڸ���
    // net�Ķ����ļ���prototxt�ļ��ﲢû��blobs����ô�����ｫ���д���__��Ȼ����
    // ��this->blobs_.size()>0��ô����blob�Ͳ���Ҫ��ʼ����,skip;��֮,����г�ʼ��  
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
    
        // ������Ϊblobs_[0]��ά��ΪN_*K_����ͨ�������ǽ�Ȩֵ������ΪN*Kά������
        // ��ô��Ϊ��������ʵ���ϣ���C++�����ݶ��Ǵ�����ڴ��У���û����ν�ľ���ĸ���
        this->blobs_[0].reset(new Blob<Dtype>(weight_shape));

        // fill the weights ������һ������ָ��
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

    // param_propagate_down_�Ǵ�Layer<Dtype> �̳��������ݳ�Ա  
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

    // ��axis=1,��M_��ʾbottom[0]�����������  
    // The first "axis" dimensions are independent inner products; the total
    // number of these is M_, the product over these dimensions.
    M_ = bottom[0]->count(0, axis);
    
    // The top shape will be the bottom shape with the flattened axes dropped,
    // and replaced by a single axis with dimension num_output (N_).
    vector<int> top_shape = bottom[0]->shape();

    // ����top_shape������ȫ���Ӳ����top��������Ҫ��bottom������ά��NxCxHxW����
    // �������á����axis=1,��ôtop������Ϊ��ά�ģ���һ������ע��vector��
    // resize����__��������£�axis֮ǰ��Ԫ�ر��ֲ���  
    top_shape.resize(axis + 1);

    // Ϊ����ĵڶ�ά��ֵ�������������;���������ΪM_  
    top_shape[axis] = N_;

    // top[0]��shape�����M_x N_  
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
forwardʵ�ֵĹ��ܾ��� y=xw'+b  
xΪ���룬ά�� MxK  
yΪ�����ά�� Nx1  
wΪȨ�أ�ά�� NxK  
bΪƫ�ã�ά�� Nx1  
���嵽����ʵ�֣��õ����������caffe_cpu_gemm������ĺ���ͷΪ:  
void caffe_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,  
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,  
    const float alpha, const float* A, const float* B, const float beta,  
    float* C)  
�������Ĺ�����ʵ��ֱ�ۣ���C����* op(A)��op(B)+��*C  
const CBLAS_TRANSPOSE TransA  # A�Ƿ�ת��  
const CBLAS_TRANSPOSE TransB  # B�Ƿ�ת��  
��TransA = CblasNoTrans, op( A ) = A����TransA = CblasTrans, op( A ) = A'  
M N K���˾���Ϊ��  
const int M <strong>//op()���������A������������C������ op()����һ��Ϊת�û���ת��  
const int N <strong>//op()���������B������������C������ 
const int K <strong>//op()���������A������������B������
������Aά����MxK��Bά����KxN��Cά��ΪMxN  
lda��ldb��ldc����BLAS���ĵ�������������ֱ�ΪABC������������ʵ��ʹ�÷��֣���
CBLAS��Ӧ����������ע���Ǿ���op()�����ľ���ABC������  

ȫ���Ӳ��forward����������:  
# ��һ����ʾ y��wx������˵��y��xw'  
caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,  
      bottom_data, weight, (Dtype)0., top_data);  
# ��һ����ʾ y��y+b  
caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,  
        bias_multiplier_.cpu_data(),  
        this->blobs_[1]->cpu_data(), (Dtype)1., top_data);<pre code_snippet_id="1584186" snippet_file_name="blog_20160221_4_1082892" name="code" class="cpp">ʵ���ϲ���ļ���Ϊ��(Mx1) x (1xN) = MxN   ��C++�У����ݶ��Ǵ洢���ڴ��У�����ָ��ָ����ôΪʲôһ����<span style="font-family: Arial, Helvetica, sans-serif;">Mx1��һ����1xN, ���˾�����Ӧ���Ǵ�����������ھ���������ԣ��������ɵ㡪����һά����Ͷ�ά����Ϊ����һά������������ڴ��еĴ洢��ʽ�ȶ�ά�����Ҫ���ɵ㣬Լ���ٵ㣬��Ϊ��ά����Ҫ����RowMajor</span>  
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
�ο�UFLDL�ϵĹ�ʽ  
��һ��������w����Ӧ�����ǣ�  
caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,  
        top_diff, bottom_data, (Dtype)0., this->blobs_[0]->mutable_cpu_diff());  
        
���չ�ʽ���У�  
��Ҫ���µ�w���ݶȵ�ά����NxK  
��ʽ�е�a^(l)��Ӧ����bottom_data��ά����MxK  
��ʽ�е�\delta_(l+1)��Ӧ����top_diff��ά����MxN  
  
�ڶ���������b����Ӧ�����ǣ�  
caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,  
        bias_multiplier_.cpu_data(), (Dtype)0.,  
        this->blobs_[1]->mutable_cpu_diff());  
���չ�ʽ���У�  
��ʽ�У�b���ݶȵ�ά��Ӧ��ΪNx1 ; \delta_(l+1)��Ӧ����top_diff��ά����MxN  
�����õ���caffe_cpu_gemv������˵�������caffe_cpu_gemm���ƣ�����ǰ���Ǽ������
������֮��ĳ˷��ģ���Ӣ���������Էֱ棬v for vector, m for matrix��������ͷ��  
void caffe_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,  
    const int N, const float alpha, const float* A, const float* x,  
    const float beta, float* y)   
# <strong>ʵ�ֵĹ������� Y����AX + ��Y������Ҫת�ã���Y����A'X + ��Y.���Ը�����
Ϊablas_sgemv()�в���MN��ʾ����op()����֮ǰ��ʱ��������������������������
Y����AX + ��Y����Y����A'X + ��Y�����Ǿ���A�����������������������ת�á�  
# ����A��ά��Ϊ MxN  
# X��һ��������ά��Ϊ Mx1  
# Y�ǽ�� ��Ҳ��һ��������ά��ΪNx1</strong>  
const CBLAS_TRANSPOSE TransA  # �Ƿ��A����ת��  
# ����Ĳ�����ֱ�ۣ���������  
const int M  
const int N  
const float alpha  
const float* A  
const float* x  
const float beta  
float* y  
�ƻص�����Ĵ���ʵ�֡�����θ���b�����ݹ�ʽb���ݶ�ֱ�Ӿ���delta  
# ���Զ�Ӧ�Ĵ�����ʵ���ǽ�top_diffת�ú�Ϳ����ˣ����Գ���bias_multiplier�ⲽ��  
caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,  
        bias_multiplier_.cpu_data(), (Dtype)0.,  
        this->blobs_[1]->mutable_cpu_diff());  
���еļ���ʵ��Ϊ��(MxN)' x (Mx1) = N x 1  
  
�������Ǽ���\delta^(l)��  
<strong>�ڹ�ʽ����һ��f������������Ժ��Ե����һ��f������Ϊ��caffeʵ���У�����
��Relu layer��ʵ�ֵģ�����ֻ��Ҫʵ������������ۼӾͺ��ˣ�����ۼ���ʵ���Եȼ�
�ھ���˷���  
caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,  
        top_diff, this->blobs_[0]->cpu_data(), (Dtype)0.,  
        (*bottom)[0]->mutable_cpu_diff());  
# top_diffΪ\delta^(l+1) ά�� MxN  
# this->blobs_[0]->cpu_data()ΪW^(l) ά�� NxK  
# (*bottom)[0]->mutable_cpu_diff()��Ҫ����Ľ����Ҳ����\delta^(l) ά����MxK  
#������ǰ���\delta^(l) ά����MxK����һ���\delta^(l+1) ά����MxN  
 
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
