#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
// io.cpp 主要定义了一些读取图像或者文件，以及它们之间的一些转化的函数。
const int kProtoReadBytesLimit = INT_MAX;  // Max size of 2 GB minus 1 byte.

namespace caffe {

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

/*
最后数据持久化函数由Protocol Buffers的工具实现，详见io.hpp
// in io.hpp
bool ReadProtoFromTextFile(const char* filename, Message* proto);
bool ReadProtoFromBinaryFile(const char* filename, Message* proto);
void WriteProtoToTextFile(const Message& proto, const char* filename);
void WriteProtoToBinaryFile(const Message& proto, const char* filename);
其中，数据可以text和binary两种格式被持久化。
*/

// 从文件读取Proto的txt文件  
bool ReadProtoFromTextFile(const char* filename, Message* proto) {
//打开文件
  int fd = open(filename, O_RDONLY);  
  CHECK_NE(fd, -1) << "File not found: " << filename;  
  //新建一个FileInputStream对象 input
  FileInputStream* input = new FileInputStream(fd);  

  //解析input文件中的Message， 即使文件中参数定义顺序与Message中的参数定义顺序不一致，也可以解析。
  // 注意如何使用protobuf去读取  
  bool success = google::protobuf::TextFormat::Parse(input, proto);  
  delete input;  
  close(fd);  
  return success;  
}  
// 和ReadProtoFromTextFile功能相反  
// 将proto写入到txt文件  
void WriteProtoToTextFile(const Message& proto, const char* filename) {  
  int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);  
  FileOutputStream* output = new FileOutputStream(fd);  
  // 注意如何写入  
  CHECK(google::protobuf::TextFormat::Print(proto, output));  
  delete output;  
  close(fd);  
}  
// 从二进制文件中读取message 参数  
// 从bin读取proto的定义  
bool ReadProtoFromBinaryFile(const char* filename, Message* proto) {  
//读取二进制文件
  int fd = open(filename, O_RDONLY);  
  CHECK_NE(fd, -1) << "File not found: " << filename;  
  ZeroCopyInputStream* raw_input = new FileInputStream(fd);  
  //  解码流com.google.protobuf.CodedInputStream  
  CodedInputStream* coded_input = new CodedInputStream(raw_input);  
  // 建立CodedInputStream类的对象coded_input
  coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);  
  //折设置最大字节限制
  bool success = proto->ParseFromCodedStream(coded_input);  
  
  delete coded_input;  
  delete raw_input;  
  close(fd);  
  return success;  
}  
//和ReadProtoFromBinaryFile功能相反 
// 将proto写入到bin文件  
void WriteProtoToBinaryFile(const Message& proto, const char* filename) {  
  fstream output(filename, ios::out | ios::trunc | ios::binary);  
  CHECK(proto.SerializeToOstream(&output));  
}  
// 以cvMat格式读入图像  
#ifdef USE_OPENCV  
// 将图像读取到CVMat，指定图像大小，是否彩色  
cv::Mat ReadImageToCVMat(const string& filename,  //is_color 为1读入彩色图像，0灰度图
    const int height, const int width, const bool is_color) {  
    //height，width都不为0则把图像resize 到height*width
  cv::Mat cv_img;  
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :  
    CV_LOAD_IMAGE_GRAYSCALE);  
  cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);  //读入图像
  if (!cv_img_origin.data) {  
    LOG(ERROR) << "Could not open or find file " << filename;  
    return cv_img_origin;  
  }  
  if (height > 0 && width > 0) {  
    cv::resize(cv_img_origin, cv_img, cv::Size(width, height));  
  } else {  
    cv_img = cv_img_origin;  
  }  
  return cv_img;  
}  
// 重载函数，提供各种不同的功能 
//重载函数，读入彩色图
cv::Mat ReadImageToCVMat(const string& filename,  
    const int height, const int width) {  
  return ReadImageToCVMat(filename, height, width, true);  
}  
  //重载函数，读入图像但不resize
cv::Mat ReadImageToCVMat(const string& filename,  
    const bool is_color) {  
  return ReadImageToCVMat(filename, 0, 0, is_color);  
}  
 //重载函数，读入彩色图像且不resize 
cv::Mat ReadImageToCVMat(const string& filename) {  
  return ReadImageToCVMat(filename, 0, 0, true);  
}  

// 匹配文件后缀名  
// Do the file extension and encoding match?  
// 看看是不是jpg还是jpeg的图像  
static bool matchExt(const std::string & fn,  
                     std::string en) {  
  //p 为文件名中“.”所在位置的索引
  size_t p = fn.rfind('.');  

  //ext为文件后缀名".xxx"
  std::string ext = p != fn.npos ? fn.substr(p) : fn;  
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);  

    //把ext中的大写字母转化小写字母
  std::transform(en.begin(), en.end(), en.begin(), ::tolower);  
  if ( ext == en )  
    return true;  
  if ( en == "jpg" && ext == "jpeg" )  
    return true;  
  return false;  
}  
// 从图像文件读取数据到Datum  
bool ReadImageToDatum(const string& filename, const int label,  
    const int height, const int width, const bool is_color,  
    const std::string & encoding, Datum* datum) {  
  cv::Mat cv_img = ReadImageToCVMat(filename, height, width, is_color);  
  if (cv_img.data) {  
    if (encoding.size()) {  
      if ( (cv_img.channels() == 3) == is_color && !height && !width &&  
          matchExt(filename, encoding) )  
        return ReadFileToDatum(filename, label, datum);  
      std::vector<uchar> buf;  
      // 对数据解码  
      cv::imencode("."+encoding, cv_img, buf);  
      datum->set_data(std::string(reinterpret_cast<char*>(&buf[0]),  
                      buf.size()));  
      // 数据标签  
      datum->set_label(label);  
      // 是否被编码  
      datum->set_encoded(true);  
      return true;  
    }  
    //cvmat转为Datum格式
    CVMatToDatum(cv_img, datum);  
    datum->set_label(label);  
    return true;  
  } else {  
    return false;  
  }  
}  
#endif  // USE_OPENCV  
// 从文件读取数据到Datum  
bool ReadFileToDatum(const string& filename, const int label,  
    Datum* datum) { 
    //获取文件指针位置 size
  std::streampos size;  
  
  fstream file(filename.c_str(), ios::in|ios::binary|ios::ate);  
  if (file.is_open()) {  
    //代表当前get 流指针的位置
    size = file.tellg();  
    std::string buffer(size, ' ');
    //设置0输入文件流的起始位置
    file.seekg(0, ios::beg);  
    file.read(&buffer[0], size);  
    file.close();  
    datum->set_data(buffer);  
    datum->set_label(label);  
    datum->set_encoded(true);  
    return true;  
  } else {  
    return false;  
  }  
}  
  
#ifdef USE_OPENCV  
// 直接编码数据的Datum到CVMat  
cv::Mat DecodeDatumToCVMatNative(const Datum& datum) {  
  cv::Mat cv_img;  
  CHECK(datum.encoded()) << "Datum not encoded";  
  const string& data = datum.data();  
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());  
  cv_img = cv::imdecode(vec_data, -1);//flag=-1  
  if (!cv_img.data) {  
    LOG(ERROR) << "Could not decode datum ";  
  }  
  return cv_img;  
}  
  
// 直接编码彩色或者非彩色Datum到CVMat  
cv::Mat DecodeDatumToCVMat(const Datum& datum, bool is_color) {  
  cv::Mat cv_img;  
  CHECK(datum.encoded()) << "Datum not encoded";  
  const string& data = datum.data();  
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());  
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :  
    CV_LOAD_IMAGE_GRAYSCALE);  

    //从内存读入图片
  cv_img = cv::imdecode(vec_data, cv_read_flag);// flag为用户指定的  
  if (!cv_img.data) {  
    LOG(ERROR) << "Could not decode datum ";  
  }  
  //将encode 的Datum转化为cvMat
  return cv_img;  
}  
  
// If Datum is encoded will decoded using DecodeDatumToCVMat and CVMatToDatum  
// If Datum is not encoded will do nothing  
bool DecodeDatumNative(Datum* datum) {  
  if (datum->encoded()) {  
    cv::Mat cv_img = DecodeDatumToCVMatNative((*datum));  
    CVMatToDatum(cv_img, datum);  
    return true;  
  } else {  
    return false;  
  }  
}  
//将encodedDatum转化为没有encode的Datum  
// 将Datum进行解码  
bool DecodeDatum(Datum* datum, bool is_color) {  
  if (datum->encoded()) {  
    cv::Mat cv_img = DecodeDatumToCVMat((*datum), is_color);  
    CVMatToDatum(cv_img, datum);  
    return true;  
  } else {  
    return false;  
  }  
}  
  
// 将CVMat转换到Datum  
void CVMatToDatum(const cv::Mat& cv_img, Datum* datum) {  
  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";  
  //分别设置channel， height，width
  datum->set_channels(cv_img.channels());  
  datum->set_height(cv_img.rows);  
  datum->set_width(cv_img.cols);  
  datum->clear_data();  
  datum->clear_float_data();  
  datum->set_encoded(false);  
  int datum_channels = datum->channels();  
  int datum_height = datum->height();  
  int datum_width = datum->width();  
  int datum_size = datum_channels * datum_height * datum_width;  
  //将buffer初始化为字符''的datum_size个副本 
  std::string buffer(datum_size, ' ');  
  for (int h = 0; h < datum_height; ++h) { 
    //指向图像第h行的指针
    const uchar* ptr = cv_img.ptr<uchar>(h);  
    int img_index = 0;  
    for (int w = 0; w < datum_width; ++w) {  
      for (int c = 0; c < datum_channels; ++c) {  
        int datum_index = (c * datum_height + h) * datum_width + w;  
        buffer[datum_index] = static_cast<char>(ptr[img_index++]);  
      }  
    }  
  }  
  datum->set_data(buffer);  
}  
#endif  // USE_OPENCV  
}  // namespace caffe  
/*
四、总结
总结起来就是，DataTransformer所作的工作实际上就是crop数据，让数据减去均值，以及缩放数据。
然后就是根据数据来推断形状。此外还介绍了io的内容，里面包含了创建临时文件临时目录操作，以及从txt文件以及bin文件读取proto数据或者写入proto的数据到txt或者bin文件。
*/