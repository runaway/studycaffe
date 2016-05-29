#include <algorithm>
#include <vector>

#include "caffe/layers/softmax_layer.hpp"
#include "caffe/util/math_functions.hpp"

/*
首先需要注意的：

整个网络的参数的 gradient 的计算方法是从顶层出发向后，在 层的时候，会拿到从 得到的 也就是 ，然后需要做两个计算：首先是自己层内的参数的 gradient，比如如果是一个普通的全连通内积层，则会有参数 和 bias 参数 ，根据刚才的 Chain Rule 式子直接计算就可以了，如果 层没有参数 (例如 Softmax 或者 Softmax-Loss 层就是没有参数的。)，这一步可以省略；其次就是“向后传递”的步骤，同样地，根据 Chain Rule 我们可以计算. 即当对输入数据bottom求导时，仅仅是为了“向后传递”，这个导数并不参与该层的参数的gradient descent 参数更新，因为该层就没有参数，何须更新参数呢？
*/
namespace caffe {

template <typename Dtype>
void SoftmaxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  top[0]->ReshapeLike(*bottom[0]);
  vector<int> mult_dims(1, bottom[0]->shape(softmax_axis_));
  sum_multiplier_.Reshape(mult_dims);
  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data);
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  vector<int> scale_dims = bottom[0]->shape();
  scale_dims[softmax_axis_] = 1;
  scale_.Reshape(scale_dims);
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* scale_data = scale_.mutable_cpu_data();
  int channels = bottom[0]->shape(softmax_axis_);
  int dim = bottom[0]->count() / outer_num_;
  caffe_copy(bottom[0]->count(), bottom_data, top_data);
  // We need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  for (int i = 0; i < outer_num_; ++i) {
    //注意，inner_num_和dim的值并不相等，dim是inner_num_的倍数，为通道数channels倍
    // initialize scale_data to the first plane
    caffe_copy(inner_num_, bottom_data + i * dim, scale_data);
    for (int j = 0; j < channels; j++) {
      for (int k = 0; k < inner_num_; k++) {
        scale_data[k] = std::max(scale_data[k],
            bottom_data[i * dim + j * inner_num_ + k]);
      }
    } //在每张图像中，沿着通道轴上取最大像素值。虽然scale_data所指向内存区域可存储的数据量为outer_num_×1×inner_num_，但每次也只是更新scale_data最前面的inner_num_个元素。scale_只是用来存储中间变量。

    //在caffe_cpu_gemm函数中，从top_data所指的内存区域选取channels×inner_num_=dim个元素进行更新，其他函数类似
    // subtraction
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_,
        1, -1., sum_multiplier_.cpu_data(), scale_data, 1., top_data);
    // exponentiation
    caffe_exp<Dtype>(dim, top_data, top_data);
    // sum after exp
    caffe_cpu_gemv<Dtype>(CblasTrans, channels, inner_num_, 1.,
        top_data, sum_multiplier_.cpu_data(), 0., scale_data);
    // division
    for (int j = 0; j < channels; j++) {
      caffe_div(inner_num_, top_data, scale_data, top_data);
      //更新top_data指针
      top_data += inner_num_;
    }
  }
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  Dtype* scale_data = scale_.mutable_cpu_data();
  int channels = top[0]->shape(softmax_axis_);
  int dim = top[0]->count() / outer_num_;

  //从top_diff拷贝到bottom_diff
  caffe_copy(top[0]->count(), top_diff, bottom_diff);
  for (int i = 0; i < outer_num_; ++i) {
    // compute dot(top_diff, top_data) and subtract them from the bottom diff
    for (int k = 0; k < inner_num_; ++k) {
      scale_data[k] = caffe_cpu_strided_dot<Dtype>(channels,
          bottom_diff + i * dim + k, inner_num_,

        //因为bottom_diff是从top_diff拷贝而来，所以caffe_cpu_strided_dot的参数不用top_diff，用bottom_diff即可，而且，这样的话就可以在下面的代码("caffe_cpu_gemm<Dtype>()")中直接更新bottom_diff.
          top_data + i * dim + k, inner_num_);
    }

    //实现的非常巧妙
    // subtraction
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_, 1,
        -1., sum_multiplier_.cpu_data(), scale_data, 1., bottom_diff + i * dim);
  }
  // elementwise multiplication
  caffe_mul(top[0]->count(), bottom_diff, top_data, bottom_diff);
}


#ifdef CPU_ONLY
STUB_GPU(SoftmaxLayer);
#endif

INSTANTIATE_CLASS(SoftmaxLayer);

}  // namespace caffe
