#ifndef CAFFE_IMAGE_DATA_LAYER_HPP_
#define CAFFE_IMAGE_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/** 
 * @brief Provides data to the Net from image files. 
 * 
 * TODO(dox): thorough documentation for Forward and proto params. 
 * ��ͼ���ļ��ж�ȡ���ݣ����Ӧ�ñȽϳ��� 
 * ��һ���б��ļ���ȡͼ���·������꣬�б��ļ���·���ڲ�����������ļ���ָ�� 
 */  
template <typename Dtype>  
class ImageDataLayer : public BasePrefetchingDataLayer<Dtype> {  
 public:  
  explicit ImageDataLayer(const LayerParameter& param)  
      : BasePrefetchingDataLayer<Dtype>(param) {}  
  virtual ~ImageDataLayer();  
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,  
      const vector<Blob<Dtype>*>& top);  
  
  virtual inline const char* type() const { return "ImageData"; }  
  virtual inline int ExactNumBottomBlobs() const { return 0; }  
  virtual inline int ExactNumTopBlobs() const { return 2; }  
  
 protected:  
  shared_ptr<Caffe::RNG> prefetch_rng_;  
  // ��ͼ���������д���  
  virtual void ShuffleImages();  
  virtual void load_batch(Batch<Dtype>* batch);  
  
  // ͼ��·��������vector  
  vector<std::pair<std::string, int> > lines_;  
  // ���������ͼ��ĸ�����Ҳ���ǵ���֮���һ��ʼ��ͼ���id  
  int lines_id_;  
};  


}  // namespace caffe

#endif  // CAFFE_IMAGE_DATA_LAYER_HPP_
