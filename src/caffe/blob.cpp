#include <climits>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// reshape �ľ���ʵ��  
// ��ʱ�ķ��������ǵ��õ��µ�reshape����  
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
���ܣ��ı�һ��blob�Ĵ�С 
���裺1.����num_��channels_��height_��width_�Ĵ�С 
2.����count_��count_ = num_ * channels_ * height_ * width_; 
3.���count_��Ϊ0��������Ϊdata_��diff_����һ��ռ� 
���countΪ0���򶼳�ʼ��ΪNULL 
���룺num��channels��height��width 
�������
*/
// reshape �ľ���ʵ��  
template <typename Dtype>  
void Blob<Dtype>::Reshape(const vector<int>& shape) {  
  CHECK_LE(shape.size(), kMaxBlobAxes); //�Ƿ�С�ڹ涨�����BLOB��ά��(35ά)  
  count_ = 1;  
  shape_.resize(shape.size());// ���Ƚ���С����Ϊvector<int> shape_; ���µ���״���ݵĴ�С  
  if (!shape_data_ || shape_data_->size() < shape.size() * sizeof(int)) {  
    shape_data_.reset(new SyncedMemory(shape.size() * sizeof(int)));//  shared_ptr<SyncedMemory> shape_data_;  
  }  
  int* shape_data = static_cast<int*>(shape_data_->mutable_cpu_data());  
  for (int i = 0; i < shape.size(); ++i) {  
    // �����״�����Ƿ�Ϸ�  
    CHECK_GE(shape[i], 0);  
    CHECK_LE(shape[i], INT_MAX / count_) << "blob size exceeds INT_MAX";  
    // �������ݸ���  
    count_ *= shape[i];  
    // ����shape���µĺ;ɵ���״����  
    shape_[i] = shape[i];  
    shape_data[i] = shape[i];  
  }  
  // �ж��Ƿ���ڴ洢������  
  if (count_ > capacity_) {  
    capacity_ = count_;  
    // ���·����ڴ�  
    data_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));  
    diff_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));  
  }  
}  
  
// ��ν��reshapeʵ���Ͼͽ����Ǹ�����shape�����ݶ���  
// �ڵ��õ�ʱ���Զ�����shape�����ݾͿ��Եõ����ݣ��е�tricky  
template <typename Dtype>  
void Blob<Dtype>::Reshape(const BlobShape& shape) {  
  // ά���Ƿ�С��35  
  CHECK_LE(shape.dim_size(), kMaxBlobAxes);  
  // ������״����  
  vector<int> shape_vec(shape.dim_size());  
  for (int i = 0; i < shape.dim_size(); ++i) {  
    shape_vec[i] = shape.dim(i);  
  }  
  // �����µ�reshape����  
  Reshape(shape_vec);  
}  
/*
���ܣ�Ϊdata_��diff_ ���·���һ��ռ䣬��С����һ��blob��һ�� 
���룺Bolb���͵�other 
�������
*/
template <typename Dtype>  
void Blob<Dtype>::ReshapeLike(const Blob<Dtype>& other) {  
  Reshape(other.shape());  
}  
/*
���ܣ��򵥵Ĺ��캯�� 
���룺num��channels��height��width
*/
template <typename Dtype>  
Blob<Dtype>::Blob(const int num, const int channels, const int height,  
    const int width)  
  // capacity_ must be initialized before calling Reshape  
  // ���ɣ��ȳ�ʼ������Ϊ0��Ȼ����reshape�������ڴ���  
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
  // ���Ҳ��gpu_data��cpu_data  
  return (const int*)shape_data_->gpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_data() const {
  CHECK(data_);
  return (const Dtype*)data_->cpu_data();
}
// ���ܣ��ı�CPU������
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
���ܣ������ĸ�������ǰ��������to_cpu(),����cpu_ptr����һ������data���󣬵ڶ�������diff���� 
������ ����to_gpu(),����gpu_ptr����һ������data���󣬵ڶ�������diff����
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

// ������blob�����ݸ��Ƶ���ǰ��blob��ȥ  
template <typename Dtype>  
void Blob<Dtype>::ShareData(const Blob& other) {  
  CHECK_EQ(count_, other.count());  
  data_ = other.data();  
}  
// ������blob��diff���ݸ��Ƶ���ǰ��blob��ȥ  
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
���ܣ�����data_�����ݣ����Ǽ�ȥdiff_�����ݡ� 
���裺1.�ж�blob��λ�ã�HEAD_AT_CPU/HEAD_AT_GPU/SYNCED/UNINITIALIZED�� 
1������caffe_axpy����math_functions.cpp�����ҵ��ú�����ʵ�֣���ʵ�⺯��Ҳ�Ƿ�װ��mkl�ĺ��������������Ϊ��ʵ�������������ļ����� 
2������caffe_gpu_axpy����math_functions.cpp�����ҵ��ú�����ʵ�֣���ʵ�⺯��Ҳ�Ƿ�װ��cublas�ĺ��������������Ϊ��ʵ�������������ļ�����

CopyFrom(const Blob& source, bool copy_diff, bool reshape) 
���ܣ���source�������ݡ�copy_diff��Ϊ��־�������ǿ���data���ǿ���diff�� 
���裺1.�����GPU�� 
����ǿ���diff������cudaMemcpy������source��diff�������� 
���򿽱�data 
2.�����CPU�� 
����ǿ���diff������memcpy������source��diff�������� 
���򿽱�data
*/
  
// Update�Ǽ���data=-1 * diff + data  
template <typename Dtype>  
void Blob<Dtype>::Update() {  
  // We will perform update based on where the data is located.  
  switch (data_->head()) {  
  case SyncedMemory::HEAD_AT_CPU:  
    // perform computation on CPU  
    // axpby��alpha * x plus beta *y �������,blas�ĺ����������Ǽ���֪��  
    // template <> void caffe_axpy<float>(const int N, const float alpha, const float* X, float* Y) { cblas_saxpy(N, alpha, X, 1, Y, 1); }  
    // caffe_axpy�������Y=alpha * X + Y ������alpha=-1������  
    // �洢��ʱ���õ���mutable_cpu_data����ֹ�����̷߳���  
    caffe_axpy<Dtype>(count_, Dtype(-1),  
        static_cast<const Dtype*>(diff_->cpu_data()),  
        static_cast<Dtype*>(data_->mutable_cpu_data()));  
    break;  
  case SyncedMemory::HEAD_AT_GPU:  
  case SyncedMemory::SYNCED:  
#ifndef CPU_ONLY  
    // perform computation on GPU  
    // Y=alpha * X + Y ������alpha=-1������  
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
// ����data��L1����  
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
  
// ����diff��L1����  
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
  
// ����sum of square of data(L2����)  
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
  
// ��data���ֳ���һ������scale_factor  
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
// ��diff���ֳ���һ������sacle_factor  
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
  
// ����blob�Ƿ�shapeһ��  
template <typename Dtype>  
bool Blob<Dtype>::ShapeEquals(const BlobProto& other) {  
  // �ж��Ƿ��Ǿɵ�blob  
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
  // ������Ǿɵ�blob��ֱ���ж�  
  vector<int> other_shape(other.shape().dim_size());  
  for (int i = 0; i < other.shape().dim_size(); ++i) {  
    other_shape[i] = other.shape().dim(i);  
  }  
  return shape_ == other_shape;  
}  

/*
���ܣ���source�������ݡ�copy_diff��Ϊ��־�������ǿ���data���ǿ���diff�� 
���裺1.�����GPU�� 
����ǿ���diff������cudaMemcpy������source��diff�������� 
���򿽱�data 
2.�����CPU�� 
����ǿ���diff������memcpy������source��diff�������� 
���򿽱�data
*/
  
// �ӱ��blob���и���  
template <typename Dtype>  
void Blob<Dtype>::CopyFrom(const Blob& source, bool copy_diff, bool reshape) {  
  if (source.count() != count_ || source.shape() != shape_) {  
    if (reshape) {  
      ReshapeLike(source);// ����shape����  
    } else {  
      LOG(FATAL) << "Trying to copy blobs of different sizes.";  
    }  
  }  
  switch (Caffe::mode()) {  
  case Caffe::GPU:  
    // GPU����diff  
    if (copy_diff) {  
        // �ⶼ�� template <> void caffe_copy<float>(const int N, const float* X, float* Y) { cblas_scopy(N, X, 1, Y, 1); }  
        // ����Ҫ��BLAS��������������ƣ����Ƕ���...  
      caffe_copy(count_, source.gpu_diff(),  
          static_cast<Dtype*>(diff_->mutable_gpu_data()));  
    } else {  
      caffe_copy(count_, source.gpu_data(),  
          static_cast<Dtype*>(data_->mutable_gpu_data()));  
    }  
    break;  
  case Caffe::CPU:  
    // CPU����diff  
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
���ܣ���proto�����ݽ�������ʵ���Ƿ����л� 
���裺1.�Ȱ�blob�Ĵ�С�ı�һ�� 
2.�õ�cpu�����ݵĵ�ַ 
3.��proto�е�data����blob�е�data 
4.��proto�е�diff����blob�е�diff
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
      // ����Ǿɵ�blobֱ��ת��Ϊ�µ�blob�е�shape����  
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
    Reshape(shape);// ����shape���ݵ���ǰblob  
  } else {  
    CHECK(ShapeEquals(proto)) << "shape mismatch (reshape not set)";  
  }  
  // copy data  
  Dtype* data_vec = mutable_cpu_data();// ��ȡ��ǰ��blob���ڴ��ϵ�����ָ�룬��ָ���ǻ����  
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
    Dtype* diff_vec = mutable_cpu_diff();// ��ȡ��ǰ��diff���ڴ��ϵ�����ָ�룬��ָ���ǻ����  
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
���ܣ�������Ĳ�������prototxt�� 
���裺 
1. ������������֣�param->set_name(name_) 
2. ���������blob������ 
3. ���ڵ�i�㣺

����bottom��blob������
����top��blob������
д��proto��
*/
// BlobProto��BlobShape��protobuf����ģ�����һЩ�������Զ����ɵ�  
// mutable_shape��add_dim��clear_double_data��clear_double_diff��add_double_data  
// add_double_diff��  
// ��src/caffe/proto/caffe.proto  
template <>  
void Blob<double>::ToProto(BlobProto* proto, bool write_diff) const {  
  proto->clear_shape();  
  // ��shape  
  for (int i = 0; i < shape_.size(); ++i) {  
    proto->mutable_shape()->add_dim(shape_[i]);  
  }  
  
  proto->clear_double_data();  
  proto->clear_double_diff();  
  // ��data  
  const double* data_vec = cpu_data();  
  for (int i = 0; i < count_; ++i) {  
    proto->add_double_data(data_vec[i]);  
  }  
  // ��diff  
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

