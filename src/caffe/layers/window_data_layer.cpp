#ifdef USE_OPENCV
#include <opencv2/highgui/highgui_c.h>
#include <stdint.h>

#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/window_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

// caffe.proto > LayerParameter > WindowDataParameter
//   'source' field specifies the window_file
//   'crop_size' indicates the desired warped size

namespace caffe {

template <typename Dtype>
WindowDataLayer<Dtype>::~WindowDataLayer<Dtype>() {
  this->StopInternalThread();
}

// 读取窗口数据文件的信息,并设置各个数据结构的形状
template <typename Dtype>
void WindowDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // LayerSetUp runs through the window_file and creates two structures
  // that hold windows: one for foreground (object) windows and one
  // for background (non-object) windows. We use an overlap threshold
  // to decide which is which.

  // window_file format
  // repeated:
  //    # image_index
  //    img_path (abs path)
  //    channels
  //    height
  //    width
  //    num_windows
  //    class_index overlap x1 y1 x2 y2

  // 窗口文件的格式如下:  
  // # 图像索引(举例:# 1就表示第一个图像,注意#号与数字之间有空格)  
  // 图像的路径  
  // 图像通道数  
  // 图像高度  
  // 图像宽度  
  // 窗口数目  
  // 类标,overlap,x1,y1,x2,y2  
  // 注:x1,y1,x2,y2是窗口的左上和右下的坐标  
  // 我这里举个例子  
  // # 1 /1.jpg 3 720 480 100 1 1 0 0 100 100  
  // 上述的例子即使表示一个编号为1的图像相对路径为/1.jpg，通道为3，高度为720  
  // 宽度为480，窗口数目为100，类标为1，overlap为1，窗口的左上坐标为(0,0),右下坐标为(100,100)  
  LOG(INFO) << "Window data layer:" << std::endl
      << "  foreground (object) overlap threshold: "
      << this->layer_param_.window_data_param().fg_threshold() << std::endl
      << "  background (non-object) overlap threshold: "
      << this->layer_param_.window_data_param().bg_threshold() << std::endl
      << "  foreground sampling fraction: "
      << this->layer_param_.window_data_param().fg_fraction() << std::endl
      << "  cache_images: "
      << this->layer_param_.window_data_param().cache_images() << std::endl
      << "  root_folder: "
      << this->layer_param_.window_data_param().root_folder();

  cache_images_ = this->layer_param_.window_data_param().cache_images();
  string root_folder = this->layer_param_.window_data_param().root_folder();

  // 根据参数文件中是否需要进行左右mirror，或者是否进行crop，  
  // 来判断是否需要初始化随机数种子  
  const bool prefetch_needs_rand =
      this->transform_param_.mirror() ||
      this->transform_param_.crop_size();
  if (prefetch_needs_rand) {
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  } else {
    prefetch_rng_.reset();
  }

  // 打开窗口文件 
  std::ifstream infile(this->layer_param_.window_data_param().source().c_str());
  CHECK(infile.good()) << "Failed to open window file "
      << this->layer_param_.window_data_param().source() << std::endl;

  // 这个是类标与类标出现的次数之间的映射  
  // 这里称之为类标直方图 
  map<int, int> label_hist;
  label_hist.insert(std::make_pair(0, 0));

  string hashtag;
  int image_index, channels;

  // 先从窗口文件中读取一个图像索引测试一下是否为空
  if (!(infile >> hashtag >> image_index)) {
    LOG(FATAL) << "Window file is empty";
  }
  do {
    // 检查是否# 开头  
    CHECK_EQ(hashtag, "#");
    // read image path
    string image_path;
    // 接下来读取图像的相对路径  
    // 将该路径与根目录路径拼接
    infile >> image_path;
    image_path = root_folder + image_path;
    // read image dimensions
    vector<int> image_size(3);
        // 读取图像的维度信息，分别为channel，height , width  
    infile >> image_size[0] >> image_size[1] >> image_size[2];
    channels = image_size[0];
        // 将图像路径和图像大小压入到image_database_中
    image_database_.push_back(std::make_pair(image_path, image_size));
    // 如果需要缓存图像到内存的话，则用image_database_cache_进行存储 
    if (cache_images_) {
      Datum datum;
            // 将图像数据读取到Datum这个结构 
      if (!ReadFileToDatum(image_path, &datum)) {
        LOG(ERROR) << "Could not open or find file " << image_path;
        return;
      }
            // 将Datum结构的图像缓存到到image_database_cache_ 
      image_database_cache_.push_back(std::make_pair(image_path, datum));
    }
    // read each box
    int num_windows;
        // 读取窗口个数 
    infile >> num_windows;
            // 从参数文件获取前景和背景阈值  
    const float fg_threshold =
        this->layer_param_.window_data_param().fg_threshold();
    const float bg_threshold =
        this->layer_param_.window_data_param().bg_threshold();
    for (int i = 0; i < num_windows; ++i) {
      int label, x1, y1, x2, y2;
      float overlap;
            // 读取  类标,与前景目标的重叠率,x1,y1,x2,y2  
      infile >> label >> overlap >> x1 >> y1 >> x2 >> y2;
      // 按照顺序放在window这个数据结构里头 
      vector<float> window(WindowDataLayer::NUM);
      window[WindowDataLayer::IMAGE_INDEX] = image_index;
      window[WindowDataLayer::LABEL] = label;
      window[WindowDataLayer::OVERLAP] = overlap;
      window[WindowDataLayer::X1] = x1;
      window[WindowDataLayer::Y1] = y1;
      window[WindowDataLayer::X2] = x2;
      window[WindowDataLayer::Y2] = y2;

      // add window to foreground list or background list
            // 下面是将窗口的前景和背景都装入到fg_windows_和bg_windows_中去  
      // 如果重叠的比例大于前景阈值，那么就认为是前景  
      if (overlap >= fg_threshold) {
        int label = window[WindowDataLayer::LABEL];
                // 类标必须大于0，因为重叠区域已经大于前景阈值了  
        // 此时如果类标不大于0，表明数据有误! 
        CHECK_GT(label, 0);
        fg_windows_.push_back(window);
                // 该类的直方图+1
        label_hist.insert(std::make_pair(label, 0));
        label_hist[label]++;
      } else if (overlap < bg_threshold) {
      // 如果重叠阈值小于背景阈值则认为是背景  
        // background window, force label and overlap to 0
        window[WindowDataLayer::LABEL] = 0;
        window[WindowDataLayer::OVERLAP] = 0;
        bg_windows_.push_back(window);
        // 0类的直方图(也就是背景的直方图)+1  
        label_hist[0]++;
      }
    }

// 每处理100个就显示一下 
    if (image_index % 100 == 0) {
      LOG(INFO) << "num: " << image_index << " "
          << image_path << " "
          << image_size[0] << " "
          << image_size[1] << " "
          << image_size[2] << " "
          << "windows to process: " << num_windows;
    }
  } while (infile >> hashtag >> image_index);
  // 读取完毕后输出图像的个数
  LOG(INFO) << "Number of images: " << image_index+1;
  // 输出统计的每个类别的个数 
  for (map<int, int>::iterator it = label_hist.begin();
      it != label_hist.end(); ++it) {
    LOG(INFO) << "class " << it->first << " has " << label_hist[it->first]
              << " samples";
  }

  LOG(INFO) << "Amount of context padding: "
      << this->layer_param_.window_data_param().context_pad();

  LOG(INFO) << "Crop mode: "
      << this->layer_param_.window_data_param().crop_mode();
  // 获取crop_size  
  // image
  const int crop_size = this->transform_param_.crop_size();
  CHECK_GT(crop_size, 0);
    // 获取batch_size 
  const int batch_size = this->layer_param_.window_data_param().batch_size();
      // 将top[0]设置为batch_size,channels, crop_size, crop_size大小的 
  top[0]->Reshape(batch_size, channels, crop_size, crop_size);
        // 将prefetch_中的数据形状也这么设置
  for (int i = 0; i < this->PREFETCH_COUNT; ++i)
    this->prefetch_[i].data_.Reshape(
        batch_size, channels, crop_size, crop_size);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
    // 将top[1]设置为类标大小 
  // label
  vector<int> label_shape(1, batch_size);
  top[1]->Reshape(label_shape);
    // 将prefetch_中的类标形状也这么设置 
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(label_shape);
  }
  // 是否有均值文件或者有均值  
  // data mean
  has_mean_file_ = this->transform_param_.has_mean_file();
  has_mean_values_ = this->transform_param_.mean_value_size() > 0;
  if (has_mean_file_) {
    // 有均值文件就读  
    const string& mean_file =
          this->transform_param_.mean_file();
    LOG(INFO) << "Loading mean file from: " << mean_file;
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
  }
  if (has_mean_values_) {
    // 有均值就直接从参数中获取
    CHECK(has_mean_file_ == false) <<
      "Cannot specify mean_file and mean_value at the same time";
    for (int c = 0; c < this->transform_param_.mean_value_size(); ++c) {
      mean_values_.push_back(this->transform_param_.mean_value(c));
    }
        // 检查均值是不是等于1，或者等于图像的通道数  
    // 也就是要么所有通道都使用同一个均值  
    // 要么每个通道用一个均值  
    CHECK(mean_values_.size() == 1 || mean_values_.size() == channels) <<
     "Specify either 1 mean_value or as many as channels: " << channels;
    if (channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }
}
// 随机数生成器进行初始化并生成随机数
template <typename Dtype>
unsigned int WindowDataLayer<Dtype>::PrefetchRand() {
  CHECK(prefetch_rng_);
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}

// 因为继承BasePrefetchingDataLayer所以要实现load_batch  
// 以供线程调用  
// This function is called on prefetch thread
template <typename Dtype>
void WindowDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  // At each iteration, sample N windows where N*p are foreground (object)
  // windows and N*(1-p) are background (non-object) windows
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  // top数据和类标  
  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = batch->label_.mutable_cpu_data();
  // 缩放尺度  
  const Dtype scale = this->layer_param_.window_data_param().scale();
  const int batch_size = this->layer_param_.window_data_param().batch_size();
  // 上下文填充 
  const int context_pad = this->layer_param_.window_data_param().context_pad();
  const int crop_size = this->transform_param_.crop_size();
  // 是否镜像 
  const bool mirror = this->transform_param_.mirror();
  // 前景比例 
  const float fg_fraction =
      this->layer_param_.window_data_param().fg_fraction();
  Dtype* mean = NULL;
  int mean_off = 0;
  int mean_width = 0;
  int mean_height = 0;
  // 如果有平均值文件则
  if (this->has_mean_file_) {
    mean = this->data_mean_.mutable_cpu_data();
    // 经过crop之后的平均值图像的中心
    mean_off = (this->data_mean_.width() - crop_size) / 2;
    mean_width = this->data_mean_.width();
    mean_height = this->data_mean_.height();
  }
  cv::Size cv_crop_size(crop_size, crop_size);
  // 获取crop的模式，是warp还是square
  const string& crop_mode = this->layer_param_.window_data_param().crop_mode();

  bool use_square = (crop_mode == "square") ? true : false;

  // zero out batch
  caffe_set(batch->data_.count(), Dtype(0), top_data);
// 根据前景比例获得前景图像的数目  
  const int num_fg = static_cast<int>(static_cast<float>(batch_size)
      * fg_fraction);
// 样本数量，是前景还是背景?[0]是背景[1]是前景 
  const int num_samples[2] = { batch_size - num_fg, num_fg };

  int item_id = 0;

    // 先对背景进行采样  
  // 再对前景进行采样 
  // sample from bg set then fg set
  for (int is_fg = 0; is_fg < 2; ++is_fg) {
    for (int dummy = 0; dummy < num_samples[is_fg]; ++dummy) {
      // sample a window
      timer.Start();
      // 生成一个随机数 
      const unsigned int rand_index = PrefetchRand();
            // fg_windows_和bg_windows_存储的是对应的窗口信息  
      // 在SetUp中读取的窗口数据文件的时候获得的  
      // 从该图像的若干窗口中去随机选择一个窗口  
      vector<float> window = (is_fg) ?
          fg_windows_[rand_index % fg_windows_.size()] :
          bg_windows_[rand_index % bg_windows_.size()];
      // 随机选择是否需要镜像  
      bool do_mirror = mirror && PrefetchRand() % 2;
      // 载入图像的路径以及类标  
      // load the image containing the window
      pair<std::string, vector<int> > image =
          image_database_[window[WindowDataLayer<Dtype>::IMAGE_INDEX]];
      // 读取图像
      cv::Mat cv_img;
      if (this->cache_images_) {
                  // 如果图像缓冲到内存则获得对应图像的Datum  
        pair<std::string, Datum> image_cached =
          image_database_cache_[window[WindowDataLayer<Dtype>::IMAGE_INDEX]];
                          // 将图像的Datum解码为OpenCV的Mat 
        cv_img = DecodeDatumToCVMat(image_cached.second, true);
      } else {
      // 否则直接读取
        cv_img = cv::imread(image.first, CV_LOAD_IMAGE_COLOR);
        if (!cv_img.data) {
          LOG(ERROR) << "Could not open or find file " << image.first;
          return;
        }
      }
      read_time += timer.MicroSeconds();
      timer.Start();
      const int channels = cv_img.channels();
    // 窗口坐标 
      // crop window out of image and warp it
      int x1 = window[WindowDataLayer<Dtype>::X1];
      int y1 = window[WindowDataLayer<Dtype>::Y1];
      int x2 = window[WindowDataLayer<Dtype>::X2];
      int y2 = window[WindowDataLayer<Dtype>::Y2];

      int pad_w = 0;
      int pad_h = 0;
            // context_pad也是个大小，具体什么含义，我没有具体研究  
      // 毕竟不是搞检测的  
      if (context_pad > 0 || use_square) {
        // scale factor by which to expand the original region
        // such that after warping the expanded region to crop_size x crop_size
        // there's exactly context_pad amount of padding on each side
        Dtype context_scale = static_cast<Dtype>(crop_size) /
            static_cast<Dtype>(crop_size - 2*context_pad);

        // compute the expanded region
                // 高度的一半  
        Dtype half_height = static_cast<Dtype>(y2-y1+1)/2.0;
         // 宽度的一半 
        Dtype half_width = static_cast<Dtype>(x2-x1+1)/2.0;
         // x中心 
        Dtype center_x = static_cast<Dtype>(x1) + half_width;
         // y中心  
        Dtype center_y = static_cast<Dtype>(y1) + half_height;
        if (use_square) {
            // 如果使用正方形形状则将较大的那个赋值给小的 
          if (half_height > half_width) {
            half_width = half_height;
          } else {
            half_height = half_width;
          }
        }
        // 获取经过处理之后的x1,y1,x2,y2  
        x1 = static_cast<int>(round(center_x - half_width*context_scale));
        x2 = static_cast<int>(round(center_x + half_width*context_scale));
        y1 = static_cast<int>(round(center_y - half_height*context_scale));
        y2 = static_cast<int>(round(center_y + half_height*context_scale));

        // 经过处理之后的窗口如果不在图像内部是有问题的  
        // 这里对窗口的坐标进行处理  
        // 使得窗口的左上角不超过图像的左上角  
        // 窗口的右下角不超过图像的右下角  
        // 所以这里叫clip bounds嘛  
        // the expanded region may go outside of the image
        // so we compute the clipped (expanded) region and keep track of
        // the extent beyond the image
        int unclipped_height = y2-y1+1;
        int unclipped_width = x2-x1+1;
        int pad_x1 = std::max(0, -x1);
        int pad_y1 = std::max(0, -y1);
        int pad_x2 = std::max(0, x2 - cv_img.cols + 1);
        int pad_y2 = std::max(0, y2 - cv_img.rows + 1);
        // clip bounds
        x1 = x1 + pad_x1;
        x2 = x2 - pad_x2;
        y1 = y1 + pad_y1;
        y2 = y2 - pad_y2;
        CHECK_GT(x1, -1);
        CHECK_GT(y1, -1);
        CHECK_LT(x2, cv_img.cols);
        CHECK_LT(y2, cv_img.rows);
        // 经过clip之后的高度和宽度  
        int clipped_height = y2-y1+1;
        int clipped_width = x2-x1+1;
        
        // scale_x/scale_y=crop_size除以未经clip之后的宽度/高度  
        // scale factors that would be used to warp the unclipped
        // expanded region
        Dtype scale_x =
            static_cast<Dtype>(crop_size)/static_cast<Dtype>(unclipped_width);
        Dtype scale_y =
            static_cast<Dtype>(crop_size)/static_cast<Dtype>(unclipped_height);

        // 用clip的宽度和高度乘以scale_x或者scale_y得到crop_size中的宽度和高度  
        // size to warp the clipped expanded region to
        cv_crop_size.width =
            static_cast<int>(round(static_cast<Dtype>(clipped_width)*scale_x));
        cv_crop_size.height =
            static_cast<int>(round(static_cast<Dtype>(clipped_height)*scale_y));

               // 再对pad的边界进行处理
        pad_x1 = static_cast<int>(round(static_cast<Dtype>(pad_x1)*scale_x));
        pad_x2 = static_cast<int>(round(static_cast<Dtype>(pad_x2)*scale_x));
        pad_y1 = static_cast<int>(round(static_cast<Dtype>(pad_y1)*scale_y));
        pad_y2 = static_cast<int>(round(static_cast<Dtype>(pad_y2)*scale_y));

        pad_h = pad_y1;

                // 如果需要镜像填充的部分也要镜像 
        // if we're mirroring, we mirror the padding too (to be pedantic)
        if (do_mirror) {
          pad_w = pad_x2;
        } else {
          pad_w = pad_x1;
        }
        // 确保大小是在crop_size x crop_size以内的 
        // ensure that the warped, clipped region plus the padding fits in the
        // crop_size x crop_size image (it might not due to rounding)
        if (pad_h + cv_crop_size.height > crop_size) {
          cv_crop_size.height = crop_size - pad_h;
        }
        if (pad_w + cv_crop_size.width > crop_size) {
          cv_crop_size.width = crop_size - pad_w;
        }
      }

      cv::Rect roi(x1, y1, x2-x1+1, y2-y1+1);
            // 进行crop  
      cv::Mat cv_cropped_img = cv_img(roi);
            // 使用线性插值进行缩放，缩放到cv_crop_size  
      cv::resize(cv_cropped_img, cv_cropped_img,
          cv_crop_size, 0, 0, cv::INTER_LINEAR);
 
      // horizontal flip at random
      if (do_mirror) {
                  // 对图像进行镜像 
        cv::flip(cv_cropped_img, cv_cropped_img, 1);
      }
 
      // copy the warped window into top_data
      for (int h = 0; h < cv_cropped_img.rows; ++h) {
        const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
        int img_index = 0;
        for (int w = 0; w < cv_cropped_img.cols; ++w) {
          for (int c = 0; c < channels; ++c) {
            int top_index = ((item_id * channels + c) * crop_size + h + pad_h)
                     * crop_size + w + pad_w;
            // int top_index = (c * height + h) * width + w;
            Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
            if (this->has_mean_file_) {
                // 有均值文件减去均值文件中对应的数值  
              int mean_index = (c * mean_height + h + mean_off + pad_h)
                           * mean_width + w + mean_off + pad_w;
              top_data[top_index] = (pixel - mean[mean_index]) * scale;
            } else {
              if (this->has_mean_values_) {
                {// 有均值则减去均值  
                top_data[top_index] = (pixel - this->mean_values_[c]) * scale;
              } else {
              // 像素值进行缩放
                top_data[top_index] = pixel * scale;
              }
            }
          }
        }
      }
      trans_time += timer.MicroSeconds();
      // get window label
      top_label[item_id] = window[WindowDataLayer<Dtype>::LABEL];

      #if 0
      // useful debugging code for dumping transformed windows to disk
      string file_id;
      std::stringstream ss;
      ss << PrefetchRand();
      ss >> file_id;
      std::ofstream inf((string("dump/") + file_id +
          string("_info.txt")).c_str(), std::ofstream::out);
      inf << image.first << std::endl
          << window[WindowDataLayer<Dtype>::X1]+1 << std::endl
          << window[WindowDataLayer<Dtype>::Y1]+1 << std::endl
          << window[WindowDataLayer<Dtype>::X2]+1 << std::endl
          << window[WindowDataLayer<Dtype>::Y2]+1 << std::endl
          << do_mirror << std::endl
          << top_label[item_id] << std::endl
          << is_fg << std::endl;
      inf.close();
      std::ofstream top_data_file((string("dump/") + file_id +
          string("_data.txt")).c_str(),
          std::ofstream::out | std::ofstream::binary);
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
            top_data_file.write(reinterpret_cast<char*>(
                &top_data[((item_id * channels + c) * crop_size + h)
                          * crop_size + w]),
                sizeof(Dtype));
          }
        }
      }
      top_data_file.close();
      #endif

      item_id++;
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(WindowDataLayer);
REGISTER_LAYER_CLASS(WindowData);

}  // namespace caffe
#endif  // USE_OPENCV
