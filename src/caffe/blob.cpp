#include <climits>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

/*
Blob作为Caffe的四大模块之一，负责完成CPU/GPU存储申请、同步和数据持久化映射。Caffe内部数据存储和通讯都是通过Blob来完成，Blob提供统一的存储操作接口，可用来保存训练数据、模型参数等。Blob是一个高维连续数组，批处理图像数据时通常使用4维Blob，Blob的维度可以表示为(N, K, H, W)，每个维度的意思分别是：
N: 数据的个数，例如SGD时一次mini-batch的图像个数。
K: 如果是图像，可以理解为通道数量；如果是网络中间结果，就是feature map的数量。
H, W： 如果是图像数据，可以理解为图像的高度和宽度；如果是参数数据，可以理解为滤波核的高度和宽度。
Caffe中通常只使用4维Blob完成图像应用，但是Blob完全可以合理地被用来存储任何数据，比如说学习到的参数。例如：
1000幅640*480 RGBD图像数据，其Blob形状为(1000, 4, 480, 640)。
96个大小11*11的滤波核，处理16通道的输入数据，其参数Blob的形状为(96，16，11，11)。
1000个输出，1024个输入的全连接层，其参数Blob的形状为(1000，1024)。
Blob是基础的数据结构，是用来保存学习到的参数以及网络传输过程中产生数据的类。在更高一级的Layer中Blob用下面的形式表示学习到的参数：vector<shared_ptr<Blob<Dtype> > > blobs_。


blob.hpp主要定义了一个Blob类。
首先看一下数据成员：
protected：
shared_ptr<SyncedMemory> data_; //shared_ptr应该为commom.hpp里引用的“using boost::shared_ptr”,
shared_ptr<SyncedMemory> diff_; //SyncedMemory类封装了CPU/GPU内存申请、同步和释放
shared_ptr<SyncedMemory> shape_data_;
vector<int> shape_;//shape_是Blob维度参数
int count_;//count表示Blob存储的元素个数（shape_所有元素乘积）
int capacity_;//capacity_表示当前Blob的元素个数（控制动态分配）


构造函数:
默认构造函数完成最基本的初始化，两个显示构造函数会调用Reshape函数完成data_和diff_的共享内存对象SyncedMemory的申请。
Reshape函数：
void Reshape(const vector<int>& shape);//主要完成数据成员shape_,shape_data_,count_,capacity_，data_，diff_最基本的初始化工作，主要包括内存分配，含初始化。
void Reshape(const BlobShape& shape);//特别是完成data_，diff_的共享内存对象SyncedMemory的申请。

Blob的数据访问方法：
const Dtype* cpu_data() const;
const Dtype* gpu_data() const;
Dtype* mutable_cpu_data();
Dtype* mutable_gpu_data();
diff类似。Blob定义了两种数据访问方式：const方式只读，不允许改写数据；mutable方式可改写数据（对diff_的访问也是类似的）。以cpu_data()为例，看看数据访问是怎样完成的。
//In blob.cpp
template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_data() const {
  CHECK(data_);
  return (const Dtype*)data_->cpu_data();
}//data_是指向SyncedMemory类的智能指针，所以这里调用的是SyncedMemory类的cpu_data()方法.注意两个函数同名，但是属于不同类的方法。
转向syncedmem.cpp
//In syncedmem.cpp
const void* SyncedMemory::cpu_data() {
  to_cpu();//首先完成数据同步，第一次访问时会申请存储空间
  return (const void*)cpu_ptr_;//返回内存指针--->void* cpu_ptr_;//In syncedmem.hpp内存指针 --->syncedmem.hpp里的几个数据成员如下：
}
======================SyncedMemory.hpp部分数据成员=============================
private：
void to_cpu(); //数据由显存同步到内存
void to_gpu(); //数据由内存同步到显存
void* cpu_ptr_; //内存指针
void* gpu_ptr_; //显存指针
size_t size_; //数据大小
SyncedHead head_; //当前数据状态，UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED
bool own_cpu_data_; //是否分配了内存空间
总结一下：Blob想要访问data_数据，由于Blob不关心细节，它会调用SyncedMemory的数据访问函数cpu_data()，由SyncedMemory的函数cpu_data()完成数据的同步并返回数据指针cpu_ptr_。

总结
Caffe中Blob封装了各种存储相关的操作，包括内存显存分配、同步、数据访问、数据读写磁盘等。它将作为基本数据模块被包含到Layer和Net中，后面将分析他们是如何被Layer和Net使用的。
*/

/*
为什么有的地方需要data copy ,有点地方不需要？？
首先需明确：
.gpu_data and .cpu_data are used in cases were the data is used only as input and will not be modified by the algorithm. .mutable_* is used when the data itself gets updated while running the algorithm.

其次，需要关注（1）对数据Blob的两次操作是否采用相同的处理器(processor),（2）之前的一次操作是否有可能更新数据Blob

Whenever a the data is called, it checks whether the previous statement was a mutable_* function call and that too using the same processor (gpu or cpu). If it is using the same processor, data need not be copied. If it is using the other processor, there is a chance that the data might have been updated in the previous .mutable_* call and hence a data copy is required.
*/
namespace caffe {

/*
修改blob的维度，如果有必要，会创建一块新的内存。该函数既可以在初始化内存时用来
创建初始内存，也可以在Layer::Reshape或Layer::Forward时用来调整top blob的维度。
当改变blob的size时，只有在当前内存不够用时才会重新创建，而且超过的内存资源不会
被释放。
注意：reshape一个输入blob后立即调用Net::Backward是错误的，应该通过Net::Forward
或Net::Reshape将新的输入的shape传递到更高层。
*/
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
// 修改blob的维度，将当前blob的维度修改成和other一样
template <typename Dtype>  
void Blob<Dtype>::ReshapeLike(const Blob<Dtype>& other) {  
  Reshape(other.shape());  
}  
/*
功能：简单的构造函数 
输入：num，channels，height，width
N: 数据的个数，例如SGD时一次mini-batch的图像个数。
K: 如果是图像，可以理解为通道数量；如果是网络中间结果，就是feature map的数量。
H, W： 如果是图像数据，可以理解为图像的高度和宽度；如果是参数数据，可以理解为滤波核的高度和宽度。

*/
template <typename Dtype>  
Blob<Dtype>::Blob(const int num, const int channels, const int height, const int width)  
  // capacity_ must be initialized before calling Reshape  
  // 技巧，先初始化容量为0，然后用reshape来分配内存了  
    : capacity_(0) 
{  
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
// 获取cpu数据指针
template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_data() const {
  CHECK(data_);
  return (const Dtype*)data_->cpu_data();
}

// 设置cpu数据指针
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

// 获取可修改的cpu数据指针
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
/*
 让当前blob的data_指向入参other的data_，这在各Layer进行Forward操作时很有用，可以进行简单的数据拷贝。
         该函数有可能会释放掉当前blob的data_，因为shared_ptr类型的data_会在使用操作符”=“进行赋值时调用其reset函数，从而调用析构函数
*/
// 将其他blob的数据复制到当前的blob中去  
template <typename Dtype>  
void Blob<Dtype>::ShareData(const Blob& other) {  
  CHECK_EQ(count_, other.count());  
  data_ = other.data();  
}  

/*
 让当前blob的diff_指向入参other的diff_，这在各Layer进行Forward操作时很有用，可以进行简单的数据拷贝。
         该函数有可能会释放掉当前blob的diff_，因为shared_ptr类型的diff_会在使用操作符”=“进行赋值时调用其reset函数，从而调用析构函数
*/
// 将其他blob的diff数据复制到当前的blob中去  
template <typename Dtype>  
void Blob<Dtype>::ShareDiff(const Blob& other) {  
  CHECK_EQ(count_, other.count());  
  diff_ = other.diff();  
}  

/*
参数更新函数----Update方法：
Blob还有一个参数更新函数也很重要Update, 它会被网络中存储参数的Blob调用，完成梯度下降过程中的参数更新。注意注释里说的“parameter blobs”，所以是针对存储参数的Blob进行参数更新。
*/

// 对数据进行计算，并更新数据
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

// 核心计算就是梯度下降更新。  
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
        static_cast<Dtype*>(data_->mutable_cpu_data()));  //调用math_function.cpp中的模板函数caffe_axpy，它封装了cblas_saxpy函数，实际上就是2个向量的相加，具体网址https://developer.apple.com/library/mac/documentation/Accelerate/Reference/BLAS_Ref/#//apple_ref/c/func/cblas_saxpy
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
// 计算blob中各数据的绝对值之和
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
// 计算blob中各差值的绝对值之和
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
// 计算blob中各数据的平方之和  
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
//  计算blob中各差值的平方之和  
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
// 用一个缩放因子对blob中的各数据进行缩放  
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
// 用一个缩放因子对blob中的各差值进行缩放  
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

从一个源blob中拷贝数据
参数source:要拷贝数据的源Blob
参数copy_diff：如果false，copy数据；如果true拷贝差值
参数reshape：如果false,要求当前blob和源blob的shape一致；
             如果true,在shape不一致的情况下会自动reshape当前的blob
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
//FromProto将BlobProto的shape,data,diff分别copy到Blob的shape_,data_,diff_,完成数据解析。
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
Blob的数据持久化函数：
Blob中存储了网络的中间处理结果和网络的参数，这些数据最终是要被存储到磁盘或从磁盘读入内存的，最后来看Blob的数据持久化函数是如何完成数据读写磁盘的。Caffe就是借助Google Protocol Buffers这个数据序列化和持久化库来完成的。
*/

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

  // 调用Blob自己的cpu_data方法获取data_,然后拷贝
  // 存data  
  const double* data_vec = cpu_data();  
  for (int i = 0; i < count_; ++i) {  
    proto->add_double_data(data_vec[i]);  
  }  
  // 存diff  
  if (write_diff) {  
    //调用Blob自己的cpu_diff方法获取diff_,然后拷贝
    const double* diff_vec = cpu_diff();  
    for (int i = 0; i < count_; ++i) {  
      proto->add_double_diff(diff_vec[i]);  
    }  
  }  //ToProto将Blob的shape_,data_,diff_分别copy到BlobProto的shape,data,diff,完成序列化.
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

