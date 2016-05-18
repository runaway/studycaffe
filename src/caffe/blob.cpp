#include <climits>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// reshape 的具体实现  
// 过时的方法最终是调用的新的reshape方法  
template <typename Dtype>  
void Blob<Dtype>::Reshape(const int num, const int channels, const int height,  
    const int width) {  
  vector<int> shape(4);  
  shape[0] = num;  
  shape[1] = channels;  
  shape[2] = height;  
  shape[3] = width;  
  Reshape(shape);  
}  
/*
功能：改变一个blob的大小 
步骤：1.读入num_，channels_，height_，width_的大小 
2.计算count_：count_ = num_ * channels_ * height_ * width_; 
3.如果count_不为0，则重新为data_和diff_分配一块空间 
如果count为0，则都初始化为NULL 
输入：num，channels，height，width 
输出：无
*/
// reshape 的具体实现  
template <typename Dtype>  
void Blob<Dtype>::Reshape(const vector<int>& shape) {  
  CHECK_LE(shape.size(), kMaxBlobAxes); //是否小于规定的最大BLOB的维度(35维)  
  count_ = 1;  
  shape_.resize(shape.size());// 首先将大小设置为vector<int> shape_; 即新的形状数据的大小  
  if (!shape_data_ || shape_data_->size() < shape.size() * sizeof(int)) {  
    shape_data_.reset(new SyncedMemory(shape.size() * sizeof(int)));//  shared_ptr<SyncedMemory> shape_data_;  
  }  
  int* shape_data = static_cast<int*>(shape_data_->mutable_cpu_data());  
  for (int i = 0; i < shape.size(); ++i) {  
    // 检查形状数据是否合法  
    CHECK_GE(shape[i], 0);  
    CHECK_LE(shape[i], INT_MAX / count_) << "blob size exceeds INT_MAX";  
    // 计算数据个数  
    count_ *= shape[i];  
    // 复制shape到新的和旧的形状数据  
    shape_[i] = shape[i];  
    shape_data[i] = shape[i];  
  }  
  // 判断是否大于存储的容量  
  if (count_ > capacity_) {  
    capacity_ = count_;  
    // 重新分配内存  
    data_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));  
    diff_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));  
  }  
}  
  
// 所谓的reshape实际上就仅仅是复制了shape的数据而已  
// 在调用的时候自动乘以shape的数据就可以得到数据，有点tricky  
template <typename Dtype>  
void Blob<Dtype>::Reshape(const BlobShape& shape) {  
  // 维度是否小于35  
  CHECK_LE(shape.dim_size(), kMaxBlobAxes);  
  // 复制形状数据  
  vector<int> shape_vec(shape.dim_size());  
  for (int i = 0; i < shape.dim_size(); ++i) {  
    shape_vec[i] = shape.dim(i);  
  }  
  // 调用新的reshape函数  
  Reshape(shape_vec);  
}  
/*
功能：为data_和diff_ 重新分配一块空间，大小和另一个blob的一样 
输入：Bolb类型的other 
输出：无
*/
template <typename Dtype>  
void Blob<Dtype>::ReshapeLike(const Blob<Dtype>& other) {  
  Reshape(other.shape());  
}  
/*
功能：简单的构造函数 
输入：num，channels，height，width
*/
template <typename Dtype>  
Blob<Dtype>::Blob(const int num, const int channels, const int height,  
    const int width)  
  // capacity_ must be initialized before calling Reshape  
  // 技巧，先初始化容量为0，然后用reshape来分配内存了  
  : capacity_(0) {  
  Reshape(num, channels, height, width);  
}  
  
template <typename Dtype>  
Blob<Dtype>::Blob(const vector<int>& shape)  
  // capacity_ must be initialized before calling Reshape  
  : capacity_(0) {  
  Reshape(shape);  
}  
  
template <typename Dtype>  
const int* Blob<Dtype>::gpu_shape() const {  
  CHECK(shape_data_);  
  // shared_ptr<SyncedMemory> shape_data_;  
  // 因此也分gpu_data和cpu_data  
  return (const int*)shape_data_->gpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_data() const {
  CHECK(data_);
  return (const Dtype*)data_->cpu_data();
}
// 功能：改变CPU的数据
template <typename Dtype>
void Blob<Dtype>::set_cpu_data(Dtype* data) {
  CHECK(data);
  data_->set_cpu_data(data);
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_data() const {
  CHECK(data_);
  return (const Dtype*)data_->gpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_diff() const {
  CHECK(diff_);
  return (const Dtype*)diff_->cpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_diff() const {
  CHECK(diff_);
  return (const Dtype*)diff_->gpu_data();
}
/*
功能：以上四个函数，前两个调用to_cpu(),返回cpu_ptr；第一个对于data对象，第二个对于diff对象 
后两个 调用to_gpu(),返回gpu_ptr；第一个对于data对象，第二个对于diff对象
*/

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_data() {
  CHECK(data_);
  return static_cast<Dtype*>(data_->mutable_cpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_data() {
  CHECK(data_);
  return static_cast<Dtype*>(data_->mutable_gpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_diff() {
  CHECK(diff_);
  return static_cast<Dtype*>(diff_->mutable_cpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_diff() {
  CHECK(diff_);
  return static_cast<Dtype*>(diff_->mutable_gpu_data());
}

// 将其他blob的数据复制到当前的blob中去  
template <typename Dtype>  
void Blob<Dtype>::ShareData(const Blob& other) {  
  CHECK_EQ(count_, other.count());  
  data_ = other.data();  
}  
// 将其他blob的diff数据复制到当前的blob中去  
template <typename Dtype>  
void Blob<Dtype>::ShareDiff(const Blob& other) {  
  CHECK_EQ(count_, other.count());  
  diff_ = other.diff();  
}  
  
// The "update" method is used for parameter blobs in a Net, which are stored  
// as Blob<float> or Blob<double> -- hence we do not define it for  
// Blob<int> or Blob<unsigned int>.  
template <> void Blob<unsigned int>::Update() { NOT_IMPLEMENTED; }  
template <> void Blob<int>::Update() { NOT_IMPLEMENTED; }  

/*
功能：更新data_的数据，就是减去diff_的数据。 
步骤：1.判断blob的位置（HEAD_AT_CPU/HEAD_AT_GPU/SYNCED/UNINITIALIZED） 
1）调用caffe_axpy：在math_functions.cpp可以找到该函数的实现，其实这函数也是封装了mkl的函数。这里调用是为了实现了两个向量的减法。 
2）调用caffe_gpu_axpy：在math_functions.cpp可以找到该函数的实现，其实这函数也是封装了cublas的函数。这里调用是为了实现了两个向量的减法。

CopyFrom(const Blob& source, bool copy_diff, bool reshape) 
功能：从source拷贝数据。copy_diff作为标志来区分是拷贝data还是拷贝diff。 
步骤：1.如果是GPU： 
如果是拷贝diff：调用cudaMemcpy函数将source的diff拷贝过来 
否则拷贝data 
2.如果是CPU： 
如果是拷贝diff：调用memcpy函数将source的diff拷贝过来 
否则拷贝data
*/
  
// Update是计算data=-1 * diff + data  
template <typename Dtype>  
void Blob<Dtype>::Update() {  
  // We will perform update based on where the data is located.  
  switch (data_->head()) {  
  case SyncedMemory::HEAD_AT_CPU:  
    // perform computation on CPU  
    // axpby即alpha * x plus beta *y 这个含义,blas的函数命名真是见名知意  
    // template <> void caffe_axpy<float>(const int N, const float alpha, const float* X, float* Y) { cblas_saxpy(N, alpha, X, 1, Y, 1); }  
    // caffe_axpy计算的是Y=alpha * X + Y ，其中alpha=-1了这里  
    // 存储的时候用到了mutable_cpu_data，防止其他线程访问  
    caffe_axpy<Dtype>(count_, Dtype(-1),  
        static_cast<const Dtype*>(diff_->cpu_data()),  
        static_cast<Dtype*>(data_->mutable_cpu_data()));  
    break;  
  case SyncedMemory::HEAD_AT_GPU:  
  case SyncedMemory::SYNCED:  
#ifndef CPU_ONLY  
    // perform computation on GPU  
    // Y=alpha * X + Y ，其中alpha=-1了这里  
    caffe_gpu_axpy<Dtype>(count_, Dtype(-1),  
        static_cast<const Dtype*>(diff_->gpu_data()),  
        static_cast<Dtype*>(data_->mutable_gpu_data()));  
#else  
    NO_GPU;  
#endif  
    break;  
  default:  
    LOG(FATAL) << "Syncedmem not initialized.";  
  }  
}  
  
template <> unsigned int Blob<unsigned int>::asum_data() const {  
  NOT_IMPLEMENTED;  
  return 0;  
}  
  
template <> int Blob<int>::asum_data() const {  
  NOT_IMPLEMENTED;  
  return 0;  
}  
// 计算data的L1范数  
template <typename Dtype>  
Dtype Blob<Dtype>::asum_data() const {  
  if (!data_) { return 0; }  
  switch (data_->head()) {  
  case SyncedMemory::HEAD_AT_CPU:  
    return caffe_cpu_asum(count_, cpu_data());  
  case SyncedMemory::HEAD_AT_GPU:  
  case SyncedMemory::SYNCED:  
#ifndef CPU_ONLY  
  {  
    Dtype asum;  
    caffe_gpu_asum(count_, gpu_data(), &asum);  
    return asum;  
  }  
#else  
    NO_GPU;  
#endif  
  case SyncedMemory::UNINITIALIZED:  
    return 0;  
  default:  
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();  
  }  
  return 0;  
}  
  
template <> unsigned int Blob<unsigned int>::asum_diff() const {  
  NOT_IMPLEMENTED;  
  return 0;  
}  
  
template <> int Blob<int>::asum_diff() const {  
  NOT_IMPLEMENTED;  
  return 0;  
}  
  
// 计算diff的L1范数  
template <typename Dtype>  
Dtype Blob<Dtype>::asum_diff() const {  
  if (!diff_) { return 0; }  
  switch (diff_->head()) {  
  case SyncedMemory::HEAD_AT_CPU:  
    return caffe_cpu_asum(count_, cpu_diff());  
  case SyncedMemory::HEAD_AT_GPU:  
  case SyncedMemory::SYNCED:  
#ifndef CPU_ONLY  
  {  
    Dtype asum;  
    caffe_gpu_asum(count_, gpu_diff(), &asum);  
    return asum;  
  }  
#else  
    NO_GPU;  
#endif  
  case SyncedMemory::UNINITIALIZED:  
    return 0;  
  default:  
    LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();  
  }  
  return 0;  
}  
  
template <> unsigned int Blob<unsigned int>::sumsq_data() const {  
  NOT_IMPLEMENTED;  
  return 0;  
}  
  
template <> int Blob<int>::sumsq_data() const {  
  NOT_IMPLEMENTED;  
  return 0;  
}  
  
// 计算sum of square of data(L2范数)  
template <typename Dtype>  
Dtype Blob<Dtype>::sumsq_data() const {  
  Dtype sumsq;  
  const Dtype* data;  
  if (!data_) { return 0; }  
  switch (data_->head()) {  
  case SyncedMemory::HEAD_AT_CPU:  
    data = cpu_data();  
    sumsq = caffe_cpu_dot(count_, data, data);  
    break;  
  case SyncedMemory::HEAD_AT_GPU:  
  case SyncedMemory::SYNCED:  
#ifndef CPU_ONLY  
    data = gpu_data();  
    caffe_gpu_dot(count_, data, data, &sumsq);  
#else  
    NO_GPU;  
#endif  
    break;  
  case SyncedMemory::UNINITIALIZED:  
    return 0;  
  default:  
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();  
  }  
  return sumsq;  
}  
  
template <> unsigned int Blob<unsigned int>::sumsq_diff() const {  
  NOT_IMPLEMENTED;  
  return 0;  
}  
  
template <> int Blob<int>::sumsq_diff() const {  
  NOT_IMPLEMENTED;  
  return 0;  
}  
  
// sum of square of diff  
template <typename Dtype>  
Dtype Blob<Dtype>::sumsq_diff() const {  
  Dtype sumsq;  
  const Dtype* diff;  
  if (!diff_) { return 0; }  
  switch (diff_->head()) {  
  case SyncedMemory::HEAD_AT_CPU:  
    diff = cpu_diff();  
    sumsq = caffe_cpu_dot(count_, diff, diff);  
    break;  
  case SyncedMemory::HEAD_AT_GPU:  
  case SyncedMemory::SYNCED:  
#ifndef CPU_ONLY  
    diff = gpu_diff();  
    caffe_gpu_dot(count_, diff, diff, &sumsq);  
    break;  
#else  
    NO_GPU;  
#endif  
  case SyncedMemory::UNINITIALIZED:  
    return 0;  
  default:  
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();  
  }  
  return sumsq;  
}  
  
template <> void Blob<unsigned int>::scale_data(unsigned int scale_factor) {  
  NOT_IMPLEMENTED;  
}  
  
template <> void Blob<int>::scale_data(int scale_factor) {  
  NOT_IMPLEMENTED;  
}  
  
// 将data部分乘以一个因子scale_factor  
template <typename Dtype>  
void Blob<Dtype>::scale_data(Dtype scale_factor) {  
  Dtype* data;  
  if (!data_) { return; }  
  switch (data_->head()) {  
  case SyncedMemory::HEAD_AT_CPU:  
    data = mutable_cpu_data();  
    caffe_scal(count_, scale_factor, data);  
    return;  
  case SyncedMemory::HEAD_AT_GPU:  
  case SyncedMemory::SYNCED:  
#ifndef CPU_ONLY  
    data = mutable_gpu_data();  
    caffe_gpu_scal(count_, scale_factor, data);  
    return;  
#else  
    NO_GPU;  
#endif  
  case SyncedMemory::UNINITIALIZED:  
    return;  
  default:  
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();  
  }  
}  
  
template <> void Blob<unsigned int>::scale_diff(unsigned int scale_factor) {  
  NOT_IMPLEMENTED;  
}  
  
template <> void Blob<int>::scale_diff(int scale_factor) {  
  NOT_IMPLEMENTED;  
}  
// 将diff部分乘以一个因子sacle_factor  
template <typename Dtype>  
void Blob<Dtype>::scale_diff(Dtype scale_factor) {  
  Dtype* diff;  
  if (!diff_) { return; }  
  switch (diff_->head()) {  
  case SyncedMemory::HEAD_AT_CPU:  
    diff = mutable_cpu_diff();  
    caffe_scal(count_, scale_factor, diff);  
    return;  
  case SyncedMemory::HEAD_AT_GPU:  
  case SyncedMemory::SYNCED:  
#ifndef CPU_ONLY  
    diff = mutable_gpu_diff();  
    caffe_gpu_scal(count_, scale_factor, diff);  
    return;  
#else  
    NO_GPU;  
#endif  
  case SyncedMemory::UNINITIALIZED:  
    return;  
  default:  
    LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();  
  }  
}  
  
// 两个blob是否shape一样  
template <typename Dtype>  
bool Blob<Dtype>::ShapeEquals(const BlobProto& other) {  
  // 判断是否是旧的blob  
  if (other.has_num() || other.has_channels() ||  
      other.has_height() || other.has_width()) {  
    // Using deprecated 4D Blob dimensions --  
    // shape is (num, channels, height, width).  
    // Note: we do not use the normal Blob::num(), Blob::channels(), etc.  
    // methods as these index from the beginning of the blob shape, where legacy  
    // parameter blobs were indexed from the end of the blob shape (e.g., bias  
    // Blob shape (1 x 1 x 1 x N), IP layer weight Blob shape (1 x 1 x M x N)).  
    return shape_.size() <= 4 &&  
           LegacyShape(-4) == other.num() &&  
           LegacyShape(-3) == other.channels() &&  
           LegacyShape(-2) == other.height() &&  
           LegacyShape(-1) == other.width();  
  }  
  // 如果不是旧的blob则直接判断  
  vector<int> other_shape(other.shape().dim_size());  
  for (int i = 0; i < other.shape().dim_size(); ++i) {  
    other_shape[i] = other.shape().dim(i);  
  }  
  return shape_ == other_shape;  
}  

/*
功能：从source拷贝数据。copy_diff作为标志来区分是拷贝data还是拷贝diff。 
步骤：1.如果是GPU： 
如果是拷贝diff：调用cudaMemcpy函数将source的diff拷贝过来 
否则拷贝data 
2.如果是CPU： 
如果是拷贝diff：调用memcpy函数将source的diff拷贝过来 
否则拷贝data
*/
  
// 从别的blob进行复制  
template <typename Dtype>  
void Blob<Dtype>::CopyFrom(const Blob& source, bool copy_diff, bool reshape) {  
  if (source.count() != count_ || source.shape() != shape_) {  
    if (reshape) {  
      ReshapeLike(source);// 复制shape数据  
    } else {  
      LOG(FATAL) << "Trying to copy blobs of different sizes.";  
    }  
  }  
  switch (Caffe::mode()) {  
  case Caffe::GPU:  
    // GPU复制diff  
    if (copy_diff) {  
        // 这都用 template <> void caffe_copy<float>(const int N, const float* X, float* Y) { cblas_scopy(N, X, 1, Y, 1); }  
        // 干嘛要用BLAS里面的运算来复制，真是多余...  
      caffe_copy(count_, source.gpu_diff(),  
          static_cast<Dtype*>(diff_->mutable_gpu_data()));  
    } else {  
      caffe_copy(count_, source.gpu_data(),  
          static_cast<Dtype*>(data_->mutable_gpu_data()));  
    }  
    break;  
  case Caffe::CPU:  
    // CPU复制diff  
    if (copy_diff) {  
      caffe_copy(count_, source.cpu_diff(),  
          static_cast<Dtype*>(diff_->mutable_cpu_data()));  
    } else {  
      caffe_copy(count_, source.cpu_data(),  
          static_cast<Dtype*>(data_->mutable_cpu_data()));  
    }  
    break;  
  default:  
    LOG(FATAL) << "Unknown caffe mode.";  
  }  
}  

/*
功能：从proto读数据进来，其实就是反序列化 
步骤：1.先把blob的大小改变一下 
2.得到cpu中数据的地址 
3.用proto中的data覆盖blob中的data 
4.用proto中的diff覆盖blob中的diff
*/
template <typename Dtype>  
void Blob<Dtype>::FromProto(const BlobProto& proto, bool reshape) {  
  // copy shape  
  if (reshape) {  
    vector<int> shape;  
    if (proto.has_num() || proto.has_channels() ||  
        proto.has_height() || proto.has_width()) {  
      // Using deprecated 4D Blob dimensions --  
      // shape is (num, channels, height, width).  
      // 如果是旧的blob直接转换为新的blob中的shape数据  
      shape.resize(4);  
      shape[0] = proto.num();  
      shape[1] = proto.channels();  
      shape[2] = proto.height();  
      shape[3] = proto.width();  
    } else {  
      shape.resize(proto.shape().dim_size());  
      for (int i = 0; i < proto.shape().dim_size(); ++i) {  
        shape[i] = proto.shape().dim(i);  
      }  
    }  
    Reshape(shape);// 复制shape数据到当前blob  
  } else {  
    CHECK(ShapeEquals(proto)) << "shape mismatch (reshape not set)";  
  }  
  // copy data  
  Dtype* data_vec = mutable_cpu_data();// 获取当前的blob在内存上的数据指针，该指针是互斥的  
  if (proto.double_data_size() > 0) {// data  
    CHECK_EQ(count_, proto.double_data_size());  
    for (int i = 0; i < count_; ++i) {  
      data_vec[i] = proto.double_data(i);  
    }  
  } else {  
    CHECK_EQ(count_, proto.data_size());  
    for (int i = 0; i < count_; ++i) {  
      data_vec[i] = proto.data(i);  
    }  
  }  
  // copy diff  
  if (proto.double_diff_size() > 0) {// diff  
    CHECK_EQ(count_, proto.double_diff_size());  
    Dtype* diff_vec = mutable_cpu_diff();// 获取当前的diff在内存上的数据指针，该指针是互斥的  
    for (int i = 0; i < count_; ++i) {  
      diff_vec[i] = proto.double_diff(i);  
    }  
  } else if (proto.diff_size() > 0) {  
    CHECK_EQ(count_, proto.diff_size());  
    Dtype* diff_vec = mutable_cpu_diff();  
    for (int i = 0; i < count_; ++i) {  
      diff_vec[i] = proto.diff(i);  
    }  
  }  
}  

/*
功能：把网络的参数存入prototxt中 
步骤： 
1. 设置网络的名字：param->set_name(name_) 
2. 加入输入层blob的名字 
3. 对于第i层：

加入bottom的blob的名字
加入top的blob的名字
写到proto中
*/
// BlobProto和BlobShape是protobuf定义的，其中一些函数是自动生成的  
// mutable_shape、add_dim、clear_double_data、clear_double_diff、add_double_data  
// add_double_diff等  
// 见src/caffe/proto/caffe.proto  
template <>  
void Blob<double>::ToProto(BlobProto* proto, bool write_diff) const {  
  proto->clear_shape();  
  // 存shape  
  for (int i = 0; i < shape_.size(); ++i) {  
    proto->mutable_shape()->add_dim(shape_[i]);  
  }  
  
  proto->clear_double_data();  
  proto->clear_double_diff();  
  // 存data  
  const double* data_vec = cpu_data();  
  for (int i = 0; i < count_; ++i) {  
    proto->add_double_data(data_vec[i]);  
  }  
  // 存diff  
  if (write_diff) {  
    const double* diff_vec = cpu_diff();  
    for (int i = 0; i < count_; ++i) {  
      proto->add_double_diff(diff_vec[i]);  
    }  
  }  
}  
  
template <>  
void Blob<float>::ToProto(BlobProto* proto, bool write_diff) const {  
  proto->clear_shape();  
  for (int i = 0; i < shape_.size(); ++i) {  
    proto->mutable_shape()->add_dim(shape_[i]);  
  }  
  proto->clear_data();  
  proto->clear_diff();  
  const float* data_vec = cpu_data();  
  for (int i = 0; i < count_; ++i) {  
    proto->add_data(data_vec[i]);  
  }  
  if (write_diff) {  
    const float* diff_vec = cpu_diff();  
    for (int i = 0; i < count_; ++i) {  
      proto->add_diff(diff_vec[i]);  
    }  
  }  
}  
  
INSTANTIATE_CLASS(Blob);
template class Blob<int>;
template class Blob<unsigned int>;

}  // namespace caffe

