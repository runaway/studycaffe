//
// This script converts the CIFAR dataset to the leveldb format used
// by caffe to perform classification.
// Usage:
//    convert_cifar_data input_folder output_db_file
// The CIFAR dataset could be downloaded at
//    http://www.cs.toronto.edu/~kriz/cifar.html

#include <fstream>  // NOLINT(readability/streams)
#include <string>

#include "boost/scoped_ptr.hpp"
#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "stdint.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"

/*
Caffe2����cifar10���ݼ�����lmdb��leveldb���͵�����

cifar10���ݼ���mnist���ݼ��洢��ʽ��ͬ��cifar10���ݼ��ѱ�ǩ��ͼ��������bin�ļ��ķ�ʽ�����ͬһ���ļ��ڣ����ִ�ŷ�ʽʹ��ÿ����cifar����bin�ļ��Ľṹ��ͬ������cifarת�����ݴ����mnist�Ĵ�����ӵ�ģ�黯����ΪԴ���ݶ�ȡģ�飨image_read����������lmdb��leveldb������ת���ı�����������������������ö��ŵ������caffe����db�ӿռ��У��������˴��룬����ʹ�ô������������

 

һ������ʼ

��ת��mnist���ݲ�ͬ���ǣ�cifar��û��ʹ��gflags�����н������ߣ�����Ҳû��ͨ��gflags�ĺ궨����ָ��Ҫת�����������ͣ����ǰ�ת�������Ͳ���ֱ����Ϊmain���������Ĳ��������ַ�ʽ������⣩��

��Create.sh�ļ��У�����convert_cifar_data.bin���Ϊ��

./build/examples/cifar10/convert_cifar_data.bin$DATA $EXAMPLE $DBTYPE

convert_cifar_data.bin���򣬳�����Ҫ3���������ֱ�ΪԴ����·����lmdb��leveldb���洢·����Ҫת������������lmdb or leveldb


ͷ�ļ���convert_mnist_data.cpp������

 

1��û������gflags�����н������ߣ�

2��û������leveldb��lmdb������ͷ�ļ�

3��������"boost/scoped_ptr.hpp"����ָ��ͷ�ļ�

4������"caffe/util/db.hpp"ͷ�ļ��������װ�˶�lmdb��leveldb���ݶ���Ĳ�������

 

[cpp] view plaincopy��������������
 
using caffe::Datum;  
using boost::scoped_ptr;  
using std::string;  
namespace db = caffe::db;  
�����ռ�����

 

1��û������ȫ��caffe�����ռ䣬���Ǿֲ�����������caffe�����ռ��µ��ӿռ� caffe::Datum��caffe::db

2������boost::scoped_ptr;����ָ�������ռ䣬����ָ�룬���ܹ���֤���뿪�����������Զ��ͷţ���mnist����ת�������У���������delete batch��ɾ����ʱ������ָ�ͨ������ָ������Զ�ɾ�����ڵı��������ڿ��Ƴ����ڴ�ռ�ú�ʵ�á�
*/

using caffe::Datum;
using boost::scoped_ptr;
using std::string;
namespace db = caffe::db;

const int kCIFARSize = 32;
const int kCIFARImageNBytes = 3072; //32*32=1024��RGB��ռһ���ֽ�,�о�Ӧ��Ϊuint8_t��0~255��  
const int kCIFARBatchSize = 10000;//cifar����5���ѵ���������ֳ�5��batches��ÿ��1��� 
const int kCIFARTrainBatches = 5;

void read_image(std::ifstream* file, int* label, char* buffer) {
  char label_char;
  file->read(&label_char, 1);
  //��ȡlabel_char�����ݣ�CIFAR10����Ӧ����һ�����ƽṹ������ݶԣ���label��data�������ԣ�����label��label_char�������  
  *label = label_char;//��label_char��ֵ����label  
  file->read(buffer, kCIFARImageNBytes);
  return;
}

void convert_dataset(const string& input_folder, const string& output_folder,
    const string& db_type) {
  scoped_ptr<db::DB> train_db(db::GetDB(db_type));
  train_db->Open(output_folder + "/cifar10_train_" + db_type, db::NEW);
  scoped_ptr<db::Transaction> txn(train_db->NewTransaction());
  // Data buffer
  int label;
  char str_buffer[kCIFARImageNBytes]; //�����ַ����飬һ��������Դ��һ��ͼƬ������  
  Datum datum;
  datum.set_channels(3);
  datum.set_height(kCIFARSize);
  datum.set_width(kCIFARSize);

  LOG(INFO) << "Writing Training data";
  for (int fileid = 0; fileid < kCIFARTrainBatches; ++fileid) {
    // Open files
    LOG(INFO) << "Training Batch " << fileid + 1;
    string batchFileName = input_folder + "/data_batch_"
      + caffe::format_int(fileid+1) + ".bin";
    std::ifstream data_file(batchFileName.c_str(),
        std::ios::in | std::ios::binary);
    CHECK(data_file) << "Unable to open train file #" << fileid + 1;
    //str_buffer=/data_batch_1.bin,�ȵȣ���str_buffer�Ǹ��ַ�����  
         //�Զ����ƺ�������ķ�ʽ���ļ�data/cifar10/data_batch_1.bin  
         //c_str() �� char* ��ʽ���� string �ں��ַ���  

   //��mnist��ͬ���ǣ�mnistԴ���ݼ���4���ļ���mnist��ȡ����ʱ���ֱ�����ļ���ȡ����read�������о���������mnistԴ������label���ݺ�image�����д洢�����ݲ�ͳһ��image�ļ��г��˴洢ͼ�������⣬���洢��ͼ��ṹ���ݣ���ͼ��ṹ���ݺ�ͼ�����ݶ�ȡ�ķ�ʽ��һ�������һ��漰�����С�˵�ת��������û�ж���һ��ͳһ��ͼ���ȡ��������ȡ����������image�ͱ�ǩ���ݶ��洢��ͬһ��bin�ļ��У����Կ��Զ���ͳһ��ͼƬ��ȡ����read_image����ȡԴ�������ݡ� 
    for (int itemid = 0; itemid < kCIFARBatchSize; ++itemid) {
        //����read_image������.bin�ļ���ȡ���ݣ�ͨ��ָ�븳ֵ��label��str_buffer 
      read_image(&data_file, &label, str_buffer);

        // ��ȡ�����ݸ�ֵ����ת�������ݶ���datum�������л�
      datum.set_label(label);
      datum.set_data(str_buffer, kCIFARImageNBytes);
      string out;
      CHECK(datum.SerializeToString(&out));
      txn->Put(caffe::format_int(fileid * kCIFARBatchSize + itemid, 5), out);
    }
  }
  txn->Commit();
  train_db->Close();

  LOG(INFO) << "Writing Testing data";
  scoped_ptr<db::DB> test_db(db::GetDB(db_type));
  test_db->Open(output_folder + "/cifar10_test_" + db_type, db::NEW);
  txn.reset(test_db->NewTransaction());
  // Open files
  std::ifstream data_file((input_folder + "/test_batch.bin").c_str(),
      std::ios::in | std::ios::binary);
  CHECK(data_file) << "Unable to open test file.";
  for (int itemid = 0; itemid < kCIFARBatchSize; ++itemid) {
    read_image(&data_file, &label, str_buffer);
    datum.set_label(label);
    datum.set_data(str_buffer, kCIFARImageNBytes);
    string out;
    CHECK(datum.SerializeToString(&out));
    txn->Put(caffe::format_int(itemid, 5), out);
  }
  txn->Commit();
  test_db->Close();
}

int main(int argc, char** argv) {
  if (argc != 4) {
    printf("This script converts the CIFAR dataset to the leveldb format used\n"
           "by caffe to perform classification.\n"
           "Usage:\n"
           "    convert_cifar_data input_folder output_folder db_type\n"
           "Where the input folder should contain the binary batch files.\n"
           "The CIFAR dataset could be downloaded at\n"
           "    http://www.cs.toronto.edu/~kriz/cifar.html\n"
           "You should gunzip them after downloading.\n");
  } else {
    google::InitGoogleLogging(argv[0]);
    convert_dataset(string(argv[1]), string(argv[2]), string(argv[3]));
  }
  return 0;
}
