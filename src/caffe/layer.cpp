#include <boost/thread.hpp>
#include "caffe/layer.hpp"

namespace caffe {

// template <typename Dtype>
// ≥ı ºªØª•≥‚¡ø  
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
