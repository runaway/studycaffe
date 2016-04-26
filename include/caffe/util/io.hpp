#ifndef CAFFE_UTIL_IO_H_
#define CAFFE_UTIL_IO_H_

#include <boost/filesystem.hpp>
#include <iomanip>
#include <iostream>  // NOLINT(readability/streams)
#include <string>

#include "google/protobuf/message.h"

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/format.hpp"

#ifndef CAFFE_TMP_DIR_RETRIES
#define CAFFE_TMP_DIR_RETRIES 100
#endif

namespace caffe {

using ::google::protobuf::Message;
using ::boost::filesystem::path;

inline void MakeTempDir(string* temp_dirname) {
  temp_dirname->clear();
  const path& model =
    boost::filesystem::temp_directory_path()/"caffe_test.%%%%-%%%%";
  for ( int i = 0; i < CAFFE_TMP_DIR_RETRIES; i++ ) {
    const path& dir = boost::filesystem::unique_path(model).string();
    bool done = boost::filesystem::create_directory(dir);
    if ( done ) {
      *temp_dirname = dir.string();
      return;
    }
  }
  LOG(FATAL) << "Failed to create a temporary directory.";
}

inline void MakeTempFilename(string* temp_filename) {
  static path temp_files_subpath;
  static uint64_t next_temp_file = 0;
  temp_filename->clear();
  if ( temp_files_subpath.empty() ) {
    string path_string="";
    MakeTempDir(&path_string);
    temp_files_subpath = path_string;
  }
  *temp_filename =
    (temp_files_subpath/caffe::format_int(next_temp_file++, 9)).string();
}
// 从txt读取proto的定义  
bool ReadProtoFromTextFile(const char* filename, Message* proto);  
  
// 从text读取proto的定义  
inline bool ReadProtoFromTextFile(const string& filename, Message* proto) {  
  return ReadProtoFromTextFile(filename.c_str(), proto);  
}  
// 从text读取proto的定义,只是增加了检查而已  
inline void ReadProtoFromTextFileOrDie(const char* filename, Message* proto) {  
  CHECK(ReadProtoFromTextFile(filename, proto));  
}  
// 从text读取proto的定义,只是增加了检查而已  
inline void ReadProtoFromTextFileOrDie(const string& filename, Message* proto) {  
  ReadProtoFromTextFileOrDie(filename.c_str(), proto);  
}  
// 将proto写入到txt文件  
void WriteProtoToTextFile(const Message& proto, const char* filename);  
inline void WriteProtoToTextFile(const Message& proto, const string& filename) {  
  WriteProtoToTextFile(proto, filename.c_str());  
}  
// 从bin读取proto的定义  
bool ReadProtoFromBinaryFile(const char* filename, Message* proto);  
// 从bin读取proto的定义  
inline bool ReadProtoFromBinaryFile(const string& filename, Message* proto) {  
  return ReadProtoFromBinaryFile(filename.c_str(), proto);  
}  
// 从bin读取proto的定义,只是增加了检查而已  
inline void ReadProtoFromBinaryFileOrDie(const char* filename, Message* proto) {  
  CHECK(ReadProtoFromBinaryFile(filename, proto));  
}  
// 从bin读取proto的定义,只是增加了检查而已  
inline void ReadProtoFromBinaryFileOrDie(const string& filename,  
                                         Message* proto) {  
  ReadProtoFromBinaryFileOrDie(filename.c_str(), proto);  
}  
  
// 将proto写入到bin文件  
void WriteProtoToBinaryFile(const Message& proto, const char* filename);  
// 内联函数，将proto写入到bin文件  
inline void WriteProtoToBinaryFile(  
    const Message& proto, const string& filename) {  
  WriteProtoToBinaryFile(proto, filename.c_str());  
}  
// 从文件读取数据到Datum  
bool ReadFileToDatum(const string& filename, const int label, Datum* datum);  
// 内联函数，从文件读取数据到Datum  
inline bool ReadFileToDatum(const string& filename, Datum* datum) {  
  return ReadFileToDatum(filename, -1, datum);  
}  
  
// 从图像文件读取数据到Datum  
bool ReadImageToDatum(const string& filename, const int label,  
    const int height, const int width, const bool is_color,  
    const std::string & encoding, Datum* datum);  
// 内联函数，从图像文件（彩色还是黑白？）读取数据到Datum，指定图像大小  
inline bool ReadImageToDatum(const string& filename, const int label,  
    const int height, const int width, const bool is_color, Datum* datum) {  
  return ReadImageToDatum(filename, label, height, width, is_color,  
                          "", datum);  
}  
// 内联函数，从彩色图像文件读取数据到Datum，指定图像大小  
inline bool ReadImageToDatum(const string& filename, const int label,  
    const int height, const int width, Datum* datum) {  
  return ReadImageToDatum(filename, label, height, width, true, datum);  
}  
// 内联函数，从图像文件（彩色还是黑白？）读取数据到Datum，自动获取图像大小  
inline bool ReadImageToDatum(const string& filename, const int label,  
    const bool is_color, Datum* datum) {  
  return ReadImageToDatum(filename, label, 0, 0, is_color, datum);  
}  
// 内联函数，从彩色图像文件读取数据到Datum，自动获取图像大小  
inline bool ReadImageToDatum(const string& filename, const int label,  
    Datum* datum) {  
  return ReadImageToDatum(filename, label, 0, 0, true, datum);  
}  
// 内联函数，从彩色图像文件读取数据到Datum，自动获取图像大小，指定编码格式  
inline bool ReadImageToDatum(const string& filename, const int label,  
    const std::string & encoding, Datum* datum) {  
  return ReadImageToDatum(filename, label, 0, 0, true, encoding, datum);  
}  
// 对Datum进行解码  
bool DecodeDatumNative(Datum* datum);  
// 对彩色图像的Datum进行解码  
bool DecodeDatum(Datum* datum, bool is_color);  
  
#ifdef USE_OPENCV  
// 将图像读取到CVMat，指定图像大小，是否彩色  
cv::Mat ReadImageToCVMat(const string& filename,  
    const int height, const int width, const bool is_color);  
// 将图像读取到CVMat，指定图像大小  
cv::Mat ReadImageToCVMat(const string& filename,  
    const int height, const int width);  
// 将图像读取到CVMat，指定是否彩色  
cv::Mat ReadImageToCVMat(const string& filename,  
    const bool is_color);  
// 将图像读取到CVMat  
cv::Mat ReadImageToCVMat(const string& filename);  
// 将Datum解码为为CVMat  
cv::Mat DecodeDatumToCVMatNative(const Datum& datum);  
// 将彩色图像的Datum解码为为CVMat  
cv::Mat DecodeDatumToCVMat(const Datum& datum, bool is_color);  
// 将CVMat转换为Datum  
void CVMatToDatum(const cv::Mat& cv_img, Datum* datum);  
#endif  // USE_OPENCV  
  
}  // namespace caffe  

#endif   // CAFFE_UTIL_IO_H_
