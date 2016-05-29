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
// io.cpp ��Ҫ������һЩ��ȡͼ������ļ����Լ�����֮���һЩת���ĺ�����
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
������ݳ־û�������Protocol Buffers�Ĺ���ʵ�֣����io.hpp
// in io.hpp
bool ReadProtoFromTextFile(const char* filename, Message* proto);
bool ReadProtoFromBinaryFile(const char* filename, Message* proto);
void WriteProtoToTextFile(const Message& proto, const char* filename);
void WriteProtoToBinaryFile(const Message& proto, const char* filename);
���У����ݿ���text��binary���ָ�ʽ���־û���
*/

// ���ļ���ȡProto��txt�ļ�  
bool ReadProtoFromTextFile(const char* filename, Message* proto) {
//���ļ�
  int fd = open(filename, O_RDONLY);  
  CHECK_NE(fd, -1) << "File not found: " << filename;  
  //�½�һ��FileInputStream���� input
  FileInputStream* input = new FileInputStream(fd);  

  //����input�ļ��е�Message�� ��ʹ�ļ��в�������˳����Message�еĲ�������˳��һ�£�Ҳ���Խ�����
  // ע�����ʹ��protobufȥ��ȡ  
  bool success = google::protobuf::TextFormat::Parse(input, proto);  
  delete input;  
  close(fd);  
  return success;  
}  
// ��ReadProtoFromTextFile�����෴  
// ��protoд�뵽txt�ļ�  
void WriteProtoToTextFile(const Message& proto, const char* filename) {  
  int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);  
  FileOutputStream* output = new FileOutputStream(fd);  
  // ע�����д��  
  CHECK(google::protobuf::TextFormat::Print(proto, output));  
  delete output;  
  close(fd);  
}  
// �Ӷ������ļ��ж�ȡmessage ����  
// ��bin��ȡproto�Ķ���  
bool ReadProtoFromBinaryFile(const char* filename, Message* proto) {  
//��ȡ�������ļ�
  int fd = open(filename, O_RDONLY);  
  CHECK_NE(fd, -1) << "File not found: " << filename;  
  ZeroCopyInputStream* raw_input = new FileInputStream(fd);  
  //  ������com.google.protobuf.CodedInputStream  
  CodedInputStream* coded_input = new CodedInputStream(raw_input);  
  // ����CodedInputStream��Ķ���coded_input
  coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);  
  //����������ֽ�����
  bool success = proto->ParseFromCodedStream(coded_input);  
  
  delete coded_input;  
  delete raw_input;  
  close(fd);  
  return success;  
}  
//��ReadProtoFromBinaryFile�����෴ 
// ��protoд�뵽bin�ļ�  
void WriteProtoToBinaryFile(const Message& proto, const char* filename) {  
  fstream output(filename, ios::out | ios::trunc | ios::binary);  
  CHECK(proto.SerializeToOstream(&output));  
}  
// ��cvMat��ʽ����ͼ��  
#ifdef USE_OPENCV  
// ��ͼ���ȡ��CVMat��ָ��ͼ���С���Ƿ��ɫ  
cv::Mat ReadImageToCVMat(const string& filename,  //is_color Ϊ1�����ɫͼ��0�Ҷ�ͼ
    const int height, const int width, const bool is_color) {  
    //height��width����Ϊ0���ͼ��resize ��height*width
  cv::Mat cv_img;  
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :  
    CV_LOAD_IMAGE_GRAYSCALE);  
  cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);  //����ͼ��
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
// ���غ������ṩ���ֲ�ͬ�Ĺ��� 
//���غ����������ɫͼ
cv::Mat ReadImageToCVMat(const string& filename,  
    const int height, const int width) {  
  return ReadImageToCVMat(filename, height, width, true);  
}  
  //���غ���������ͼ�񵫲�resize
cv::Mat ReadImageToCVMat(const string& filename,  
    const bool is_color) {  
  return ReadImageToCVMat(filename, 0, 0, is_color);  
}  
 //���غ����������ɫͼ���Ҳ�resize 
cv::Mat ReadImageToCVMat(const string& filename) {  
  return ReadImageToCVMat(filename, 0, 0, true);  
}  

// ƥ���ļ���׺��  
// Do the file extension and encoding match?  
// �����ǲ���jpg����jpeg��ͼ��  
static bool matchExt(const std::string & fn,  
                     std::string en) {  
  //p Ϊ�ļ����С�.������λ�õ�����
  size_t p = fn.rfind('.');  

  //extΪ�ļ���׺��".xxx"
  std::string ext = p != fn.npos ? fn.substr(p) : fn;  
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);  

    //��ext�еĴ�д��ĸת��Сд��ĸ
  std::transform(en.begin(), en.end(), en.begin(), ::tolower);  
  if ( ext == en )  
    return true;  
  if ( en == "jpg" && ext == "jpeg" )  
    return true;  
  return false;  
}  
// ��ͼ���ļ���ȡ���ݵ�Datum  
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
      // �����ݽ���  
      cv::imencode("."+encoding, cv_img, buf);  
      datum->set_data(std::string(reinterpret_cast<char*>(&buf[0]),  
                      buf.size()));  
      // ���ݱ�ǩ  
      datum->set_label(label);  
      // �Ƿ񱻱���  
      datum->set_encoded(true);  
      return true;  
    }  
    //cvmatתΪDatum��ʽ
    CVMatToDatum(cv_img, datum);  
    datum->set_label(label);  
    return true;  
  } else {  
    return false;  
  }  
}  
#endif  // USE_OPENCV  
// ���ļ���ȡ���ݵ�Datum  
bool ReadFileToDatum(const string& filename, const int label,  
    Datum* datum) { 
    //��ȡ�ļ�ָ��λ�� size
  std::streampos size;  
  
  fstream file(filename.c_str(), ios::in|ios::binary|ios::ate);  
  if (file.is_open()) {  
    //����ǰget ��ָ���λ��
    size = file.tellg();  
    std::string buffer(size, ' ');
    //����0�����ļ�������ʼλ��
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
// ֱ�ӱ������ݵ�Datum��CVMat  
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
  
// ֱ�ӱ����ɫ���߷ǲ�ɫDatum��CVMat  
cv::Mat DecodeDatumToCVMat(const Datum& datum, bool is_color) {  
  cv::Mat cv_img;  
  CHECK(datum.encoded()) << "Datum not encoded";  
  const string& data = datum.data();  
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());  
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :  
    CV_LOAD_IMAGE_GRAYSCALE);  

    //���ڴ����ͼƬ
  cv_img = cv::imdecode(vec_data, cv_read_flag);// flagΪ�û�ָ����  
  if (!cv_img.data) {  
    LOG(ERROR) << "Could not decode datum ";  
  }  
  //��encode ��Datumת��ΪcvMat
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
//��encodedDatumת��Ϊû��encode��Datum  
// ��Datum���н���  
bool DecodeDatum(Datum* datum, bool is_color) {  
  if (datum->encoded()) {  
    cv::Mat cv_img = DecodeDatumToCVMat((*datum), is_color);  
    CVMatToDatum(cv_img, datum);  
    return true;  
  } else {  
    return false;  
  }  
}  
  
// ��CVMatת����Datum  
void CVMatToDatum(const cv::Mat& cv_img, Datum* datum) {  
  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";  
  //�ֱ�����channel�� height��width
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
  //��buffer��ʼ��Ϊ�ַ�''��datum_size������ 
  std::string buffer(datum_size, ' ');  
  for (int h = 0; h < datum_height; ++h) { 
    //ָ��ͼ���h�е�ָ��
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
�ġ��ܽ�
�ܽ��������ǣ�DataTransformer�����Ĺ���ʵ���Ͼ���crop���ݣ������ݼ�ȥ��ֵ���Լ��������ݡ�
Ȼ����Ǹ����������ƶ���״�����⻹������io�����ݣ���������˴�����ʱ�ļ���ʱĿ¼�������Լ���txt�ļ��Լ�bin�ļ���ȡproto���ݻ���д��proto�����ݵ�txt����bin�ļ���
*/