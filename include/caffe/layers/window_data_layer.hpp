#ifndef CAFFE_WINDOW_DATA_LAYER_HPP_
#define CAFFE_WINDOW_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

/*
��9��WindowDataLayer��Ķ����Լ�ʵ�����£�
������Ҫ���Ƕ��ڶ�ȡ�õ�Datum����OpenCV��ȡ��Mat��Vector����Ԥ����ͼ���crop��scale�ȣ���Ȼ��ǰ����


���ȸ������������ļ��ĸ�ʽ�������Լ�ѵ��


�����ļ��ĸ�ʽ����:
# ͼ������(����:# 1�ͱ�ʾ��һ��ͼ��,ע��#��������֮���пո�)
ͼ���·��
ͼ��ͨ����
ͼ��߶�
ͼ����
������Ŀ
���,��ǰ��Ŀ����ص���,x1,y1,x2,y2
ע:x1,y1,x2,y2�Ǵ��ڵ����Ϻ����µ�����


Ϊ�����ĸ����������ٸ����ӣ�
# 1 /1.jpg 3 720 480 100 1 1 0 0 100 100 2 30 100 1500 1500
���������ӱ�ʾһ�����Ϊ1��ͼ�����·��Ϊ/1.jpg��ͨ��Ϊ3���߶�Ϊ720
���Ϊ480��������ĿΪ100�����Ϊ1����ǰ��Ŀ����ص���Ϊ0.8�����Ϊ1���ڵ���������Ϊ(0,0),��������Ϊ(100,100)
���Ϊ2�Ĵ��ڵ����Ͻ�����Ϊ(30,100)�����½ǵ�����Ϊ(1500,1500)���ж��ٴ��ں������ô����д��ȥ
*/

namespace caffe {

/**
 * @brief Provides data to the Net from windows of images files, specified
 *        by a window data file.
 * ��ͼ���ļ��Ĵ��ڻ�ȡ���ݣ���Ҫָ�����������ļ� 
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class WindowDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit WindowDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~WindowDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "WindowData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  virtual unsigned int PrefetchRand();
  virtual void load_batch(Batch<Dtype>* batch);

  shared_ptr<Caffe::RNG> prefetch_rng_;
  vector<std::pair<std::string, vector<int> > > image_database_;
  // ����������ʹ�õĴ������ݵ�ö��  
  // ���Ƕ����vector<float>��Ȼ�����水˳����������Щ���͵�����  
  enum WindowField { IMAGE_INDEX, LABEL, OVERLAP, X1, Y1, X2, Y2, NUM };
  vector<vector<float> > fg_windows_;
  vector<vector<float> > bg_windows_;
  Blob<Dtype> data_mean_;
  vector<Dtype> mean_values_;
  bool has_mean_file_;
  bool has_mean_values_;
  bool cache_images_;
  vector<std::pair<std::string, Datum > > image_database_cache_;
};

}  // namespace caffe

#endif  // CAFFE_WINDOW_DATA_LAYER_HPP_
