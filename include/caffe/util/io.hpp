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
// ��txt��ȡproto�Ķ���  
bool ReadProtoFromTextFile(const char* filename, Message* proto);  
  
// ��text��ȡproto�Ķ���  
inline bool ReadProtoFromTextFile(const string& filename, Message* proto) {  
  return ReadProtoFromTextFile(filename.c_str(), proto);  
}  
// ��text��ȡproto�Ķ���,ֻ�������˼�����  
inline void ReadProtoFromTextFileOrDie(const char* filename, Message* proto) {  
  CHECK(ReadProtoFromTextFile(filename, proto));  
}  
// ��text��ȡproto�Ķ���,ֻ�������˼�����  
inline void ReadProtoFromTextFileOrDie(const string& filename, Message* proto) {  
  ReadProtoFromTextFileOrDie(filename.c_str(), proto);  
}  
// ��protoд�뵽txt�ļ�  
void WriteProtoToTextFile(const Message& proto, const char* filename);  
inline void WriteProtoToTextFile(const Message& proto, const string& filename) {  
  WriteProtoToTextFile(proto, filename.c_str());  
}  
// ��bin��ȡproto�Ķ���  
bool ReadProtoFromBinaryFile(const char* filename, Message* proto);  
// ��bin��ȡproto�Ķ���  
inline bool ReadProtoFromBinaryFile(const string& filename, Message* proto) {  
  return ReadProtoFromBinaryFile(filename.c_str(), proto);  
}  
// ��bin��ȡproto�Ķ���,ֻ�������˼�����  
inline void ReadProtoFromBinaryFileOrDie(const char* filename, Message* proto) {  
  CHECK(ReadProtoFromBinaryFile(filename, proto));  
}  
// ��bin��ȡproto�Ķ���,ֻ�������˼�����  
inline void ReadProtoFromBinaryFileOrDie(const string& filename,  
                                         Message* proto) {  
  ReadProtoFromBinaryFileOrDie(filename.c_str(), proto);  
}  
  
// ��protoд�뵽bin�ļ�  
void WriteProtoToBinaryFile(const Message& proto, const char* filename);  
// ������������protoд�뵽bin�ļ�  
inline void WriteProtoToBinaryFile(  
    const Message& proto, const string& filename) {  
  WriteProtoToBinaryFile(proto, filename.c_str());  
}  
// ���ļ���ȡ���ݵ�Datum  
bool ReadFileToDatum(const string& filename, const int label, Datum* datum);  
// �������������ļ���ȡ���ݵ�Datum  
inline bool ReadFileToDatum(const string& filename, Datum* datum) {  
  return ReadFileToDatum(filename, -1, datum);  
}  
  
// ��ͼ���ļ���ȡ���ݵ�Datum  
bool ReadImageToDatum(const string& filename, const int label,  
    const int height, const int width, const bool is_color,  
    const std::string & encoding, Datum* datum);  
// ������������ͼ���ļ�����ɫ���Ǻڰף�����ȡ���ݵ�Datum��ָ��ͼ���С  
inline bool ReadImageToDatum(const string& filename, const int label,  
    const int height, const int width, const bool is_color, Datum* datum) {  
  return ReadImageToDatum(filename, label, height, width, is_color,  
                          "", datum);  
}  
// �����������Ӳ�ɫͼ���ļ���ȡ���ݵ�Datum��ָ��ͼ���С  
inline bool ReadImageToDatum(const string& filename, const int label,  
    const int height, const int width, Datum* datum) {  
  return ReadImageToDatum(filename, label, height, width, true, datum);  
}  
// ������������ͼ���ļ�����ɫ���Ǻڰף�����ȡ���ݵ�Datum���Զ���ȡͼ���С  
inline bool ReadImageToDatum(const string& filename, const int label,  
    const bool is_color, Datum* datum) {  
  return ReadImageToDatum(filename, label, 0, 0, is_color, datum);  
}  
// �����������Ӳ�ɫͼ���ļ���ȡ���ݵ�Datum���Զ���ȡͼ���С  
inline bool ReadImageToDatum(const string& filename, const int label,  
    Datum* datum) {  
  return ReadImageToDatum(filename, label, 0, 0, true, datum);  
}  
// �����������Ӳ�ɫͼ���ļ���ȡ���ݵ�Datum���Զ���ȡͼ���С��ָ�������ʽ  
inline bool ReadImageToDatum(const string& filename, const int label,  
    const std::string & encoding, Datum* datum) {  
  return ReadImageToDatum(filename, label, 0, 0, true, encoding, datum);  
}  
// ��Datum���н���  
bool DecodeDatumNative(Datum* datum);  
// �Բ�ɫͼ���Datum���н���  
bool DecodeDatum(Datum* datum, bool is_color);  
  
#ifdef USE_OPENCV  
// ��ͼ���ȡ��CVMat��ָ��ͼ���С���Ƿ��ɫ  
cv::Mat ReadImageToCVMat(const string& filename,  
    const int height, const int width, const bool is_color);  
// ��ͼ���ȡ��CVMat��ָ��ͼ���С  
cv::Mat ReadImageToCVMat(const string& filename,  
    const int height, const int width);  
// ��ͼ���ȡ��CVMat��ָ���Ƿ��ɫ  
cv::Mat ReadImageToCVMat(const string& filename,  
    const bool is_color);  
// ��ͼ���ȡ��CVMat  
cv::Mat ReadImageToCVMat(const string& filename);  
// ��Datum����ΪΪCVMat  
cv::Mat DecodeDatumToCVMatNative(const Datum& datum);  
// ����ɫͼ���Datum����ΪΪCVMat  
cv::Mat DecodeDatumToCVMat(const Datum& datum, bool is_color);  
// ��CVMatת��ΪDatum  
void CVMatToDatum(const cv::Mat& cv_img, Datum* datum);  
#endif  // USE_OPENCV  
  
}  // namespace caffe  

#endif   // CAFFE_UTIL_IO_H_
