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
（9）WindowDataLayer类的定义以及实现如下：
该类主要就是对于读取好的Datum或者OpenCV读取的Mat的Vector进行预处理（图像的crop、scale等），然后前传。


首先给出窗口数据文件的格式，便于自己训练


窗口文件的格式如下:
# 图像索引(举例:# 1就表示第一个图像,注意#号与数字之间有空格)
图像的路径
图像通道数
图像高度
图像宽度
窗口数目
类标,与前景目标的重叠率,x1,y1,x2,y2
注:x1,y1,x2,y2是窗口的左上和右下的坐标


为了理解的更清楚我这里举个例子：
# 1 /1.jpg 3 720 480 100 1 1 0 0 100 100 2 30 100 1500 1500
上述的例子表示一个编号为1的图像相对路径为/1.jpg，通道为3，高度为720
宽度为480，窗口数目为100，类标为1，与前景目标的重叠率为0.8，类标为1窗口的左上坐标为(0,0),右下坐标为(100,100)
类标为2的窗口的左上角坐标为(30,100)，右下角的坐标为(1500,1500)。有多少窗口后面就这么继续写下去
*/

namespace caffe {

/**
 * @brief Provides data to the Net from windows of images files, specified
 *        by a window data file.
 * 从图像文件的窗口获取数据，需要指定窗口数据文件 
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
  // 窗口类中所使用的窗口数据的枚举  
  // 就是定义个vector<float>，然后里面按顺序存放下面这些类型的数据  
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
