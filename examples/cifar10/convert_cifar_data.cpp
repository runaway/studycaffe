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
Caffe2——cifar10数据集创建lmdb或leveldb类型的数据

cifar10数据集和mnist数据集存储方式不同，cifar10数据集把标签和图像数据以bin文件的方式存放在同一个文件内，这种存放方式使得每个子cifar数据bin文件的结构相同，所以cifar转换数据代码比mnist的代码更加的模块化，分为源数据读取模块（image_read函数），把lmdb（leveldb）数据转换的变量声明，句柄（函数）调用都放到定义的caffe：：db子空间中，这样简化了代码，而且使得代码更加清晰。

 

一：程序开始

和转换mnist数据不同的是，cifar并没有使用gflags命令行解析工具；所以也没有通过gflags的宏定义来指定要转换的数据类型，而是把转换的类型参数直接作为main（）函数的参数（这种方式便于理解）。

在Create.sh文件中，调用convert_cifar_data.bin语句为：

./build/examples/cifar10/convert_cifar_data.bin$DATA $EXAMPLE $DBTYPE

convert_cifar_data.bin程序，程序需要3个参数，分别为源数据路径，lmdb（leveldb）存储路径，要转换的数据类型lmdb or leveldb


头文件和convert_mnist_data.cpp的区别：

 

1，没有引入gflags命令行解析工具；

2，没有引入leveldb和lmdb的数据头文件

3，引入了"boost/scoped_ptr.hpp"智能指针头文件

4，引入"caffe/util/db.hpp"头文件，里面包装了对lmdb和leveldb数据对象的操作内容

 

[cpp] view plaincopy技术分享技术分享
 
using caffe::Datum;  
using boost::scoped_ptr;  
using std::string;  
namespace db = caffe::db;  
命名空间区别：

 

1，没有引入全部caffe命名空间，而是局部引入了两个caffe命名空间下的子空间 caffe::Datum和caffe::db

2，引入boost::scoped_ptr;智能指针命名空间，智能指针，它能够保证在离开作用域后对象被自动释放；在mnist数据转换代码中，经常出现delete batch等删除临时变量的指令，通过智能指针可以自动删除过期的变量，对于控制程序内存占用很实用。
*/

using caffe::Datum;
using boost::scoped_ptr;
using std::string;
namespace db = caffe::db;

const int kCIFARSize = 32;
const int kCIFARImageNBytes = 3072; //32*32=1024，RGB各占一个字节,感觉应该为uint8_t，0~255，  
const int kCIFARBatchSize = 10000;//cifar共计5万个训练样本，分成5份batches，每份1万个 
const int kCIFARTrainBatches = 5;

void read_image(std::ifstream* file, int* label, char* buffer) {
  char label_char;
  file->read(&label_char, 1);
  //读取label_char的内容；CIFAR10数据应该是一个类似结构体的数据对，有label和data两个属性，其中label用label_char来定义的  
  *label = label_char;//把label_char的值，给label  
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
  char str_buffer[kCIFARImageNBytes]; //定义字符数组，一个数组可以存放一张图片的数据  
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
    //str_buffer=/data_batch_1.bin,等等，但str_buffer是个字符数组  
         //以二进制和流输入的方式打开文件data/cifar10/data_batch_1.bin  
         //c_str() 以 char* 形式传回 string 内含字符串  

   //和mnist不同的是，mnist源数据集有4个文件；mnist读取数据时，分别调用文件读取函数read（），感觉这是由于mnist源数据中label数据和image数据中存储的内容不统一，image文件中除了存储图像数据外，还存储了图像结构数据；而图像结构数据和图像数据读取的方式不一样，而且还涉及到大端小端的转换；所以没有定义一个统一的图像读取函数来读取；本项由于image和标签数据都存储在同一个bin文件中，所以可以定义统一的图片读取函数read_image来读取源数据内容。 
    for (int itemid = 0; itemid < kCIFARBatchSize; ++itemid) {
        //调用read_image函数从.bin文件读取数据，通过指针赋值给label和str_buffer 
      read_image(&data_file, &label, str_buffer);

        // 读取的数据赋值到“转换”数据对象datum，并序列化
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
