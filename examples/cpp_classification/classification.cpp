#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

/* pair(标签，置信度)  预测值 */
/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

// 分类，默认返回前5个预测值[(标签，置信度),... ] 数组
class Classifier 
{
public:
    Classifier(const string& model_file,
         const string& trained_file,
         const string& mean_file,
         const string& label_file);

    std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);

private:
    void SetMean(const string& mean_file);

    std::vector<float> Predict(const cv::Mat& img);

    void WrapInputLayer(std::vector<cv::Mat>* input_channels);

    void Preprocess(const cv::Mat& img,
              std::vector<cv::Mat>* input_channels);

private:

    shared_ptr<Net<float> > net_; // caffe分类网络对象
    cv::Size input_geometry_; // 输入数据的几何维度,宽和高 
    int num_channels_; // 通道数 
    cv::Mat mean_; // 均值图像
    std::vector<string> labels_; // 目标标签数组
};

Classifier::Classifier(const string& model_file,
                       const string& trained_file,
                       const string& mean_file,
                       const string& label_file) 
{
#ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
#else
    Caffe::set_mode(Caffe::GPU);
#endif

    /* Load the network. */
    // 加载网络拓扑结构 
    net_.reset(new Net<float>(model_file, TEST));

    // 加载网络权重 
    net_->CopyTrainedLayersFrom(trained_file);

    // 调用glog的检查 
    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";

    // 检查 
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

    // 网络层模板 Blob
    Blob<float>* input_layer = net_->input_blobs()[0];

    // 通道数
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

    // 加载均值文件
    /* Load the binaryproto mean file. */
    SetMean(mean_file);

    // 加载分类标签文件
    /* Load labels. */
    std::ifstream labels(label_file.c_str());
    CHECK(labels) << "Unable to open labels file " << label_file;
    string line;
    
    while (std::getline(labels, line))
    labels_.push_back(string(line));

    Blob<float>* output_layer = net_->output_blobs()[0];

    // 检查labels_的长度与输出层的维数是否一致 
    CHECK_EQ(labels_.size(), output_layer->channels())
    << "Number of labels is different from the output layer dimension.";
}

// 下面两个函数是排序函数代码 
static bool PairCompare(const std::pair<float, int>& lhs,
                const std::pair<float, int>& rhs) 
{
    return lhs.first > rhs.first;
}

// 返回数组v[] 最大值的前 N 个序号数组
/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) 
{
    std::vector<std::pair<float, int> > pairs;
    
    for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
    
    std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

    std::vector<int> result;
    
    for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
    
    return result;
}

// 分类并返回最大的前 N 个预测
/* Return the top N predictions. */
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N) 
{
    std::vector<float> output = Predict(img);

    N = std::min<int>(labels_.size(), N);

    // 取前N个预测结果 
    std::vector<int> maxN = Argmax(output, N);
    std::vector<Prediction> predictions;
    
    for (int i = 0; i < N; ++i) 
    {
        int idx = maxN[i];

        // 组成 [(标签，置信度),...]预测值数组
        predictions.push_back(std::make_pair(labels_[idx], output[idx]));
    }

    return predictions;
}

// 从二进制的bin文件中读取均值，并设置到blob_中
/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file) 
{
    // 调用google/protobuf?? ,用于加速运算的数据接口，有时间再详细了解其应用方法 
    BlobProto blob_proto;
    // 这个函数是实现了从二进制文件中读取数据到blob_proto中,猜测函数来自第3方库的google/protobuf模块 
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

    /* Convert from BlobProto to Blob<float> */
    Blob<float> mean_blob;

    // 调用Blob类的成员函数FromRroto从BlobProto中加载数据 
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), num_channels_)
    << "Number of channels of mean file doesn't match input layer.";

    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector<cv::Mat> channels;

    // 用可读写的方式取得指针 
    float* data = mean_blob.mutable_cpu_data();

    // 把均值上的各个通道的复制到 vector<Mat> channels，即channels[0]中对应均值中的通道0， 
    // 这样做的原因是 Blob类的数据存储方式是一维的。 
    // 我们这里是把一维度的数组 转化为Mat数组了 
    for (int i = 0; i < num_channels_; ++i) 
    {
        /* Extract an individual channel. */
        cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
        channels.push_back(channel);
        data += mean_blob.height() * mean_blob.width();
    }

    /* Merge the separate channels into a single image. */
    cv::Mat mean;

    // 合并分开的通道为一个图像，即把channels的所有Mat合并为一个Mat. 
    cv::merge(channels, mean);

    // 计算每个像素在所有通道上的平均值，保存在channel_mean中 
    /* Compute the global mean pixel value and create a mean image
    * filled with this value. */
    cv::Scalar channel_mean = cv::mean(mean);

    // 赋值给 本类的成员变量mean_ 
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

// 对图片进行预测 
std::vector<float> Classifier::Predict(const cv::Mat& img) 
{
    // 得到net的输入层数据指针 
    Blob<float>* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, num_channels_,
                   input_geometry_.height, input_geometry_.width);

    /* Forward dimension change to all layers. */
    net_->Reshape();

    std::vector<cv::Mat> input_channels;

    // 打包输入层
    // 将net_->input_blobs()[0]的地址给input_channels 
    WrapInputLayer(&input_channels);

    // 数据预处理
    // 将图片地址给input_channels 
    Preprocess(img, &input_channels);

    // 所有层前向计算 
    net_->Forward();

    // 将net的输出层数据复制到vector<float>类型的变量中，并返回 
    /* Copy the output layer to a std::vector */
    Blob<float>* output_layer = net_->output_blobs()[0];
    const float* begin = output_layer->cpu_data();
    const float* end = begin + output_layer->channels();

    return std::vector<float>(begin, end);
}

// 将net_的数据接口与input_channels 对接 
/* 打包网络中不同的的输入层 cv:Mat 对象
  （每个通道一个）。这样我们保存一个 memcpy的操作，我们
   并不需要依靠cudaMemcpy2D 。最后预处理
   操作将直接写入不同通道的输入层。
Wrap the input layer of the network in separate cv::Mat objects
* (one per channel). This way we save one memcpy operation and we
* don't need to rely on cudaMemcpy2D. The last preprocessing
* operation will write the separate channels directly to the input
* layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) 
{
    Blob<float>* input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();

    // 取出Blob类的数据，并在后续部分对齐进行修改(即在Preprocess中，将图片的值放入input_layer中) 
    float* input_data = input_layer->mutable_cpu_data();

    for (int i = 0; i < input_layer->channels(); ++i) 
    {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}

// 以img为输入，用net_来forword计算输出层值。 
void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) 
{
    // 保证输入图片的channels与网络channels一致
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;

    // 通道数据根据设置进行转换
    if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
    sample = img;
    
    // 保证大小一致 
    cv::Mat sample_resized;
    
    if (sample.size() != input_geometry_)
    {
        cv::resize(sample, sample_resized, input_geometry_);
    }
    else
    {
        sample_resized = sample;
    }
    
    // 保证数据类型一致为 float 
    cv::Mat sample_float;
    
    if (num_channels_ == 3)
    {
        sample_resized.convertTo(sample_float, CV_32FC3);
    }
    else
    {
        sample_resized.convertTo(sample_float, CV_32FC1);
    }
    
    // 减去均值得到sample_normalized 
    cv::Mat sample_normalized;
    cv::subtract(sample_float, mean_, sample_normalized);

    // 将数据BGR直接写入输入层对象input_channels
    // 将 sample_normalized 放入 input_channels中，即放入net_->input_blob中。
    /* This operation will write the separate BGR planes directly to the
    * input layer of the network because it is wrapped by the cv::Mat
    * objects in input_channels. */
    cv::split(sample_normalized, *input_channels);

    CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
    == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

int main(int argc, char** argv) 
{
    if (argc != 6) 
    {
        std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " mean.binaryproto labels.txt img.jpg" << std::endl;
        return 1;
    }
    
    // glog 库内函数，glog 库是一个做日志的库 
    ::google::InitGoogleLogging(argv[0]);

    string model_file   = argv[1]; // 网络结构文件名
    string trained_file = argv[2]; // 训练权值文件名
    string mean_file    = argv[3]; // 均值文件名
    string label_file   = argv[4]; // 分类文件名

    // 创建分类器
    Classifier classifier(model_file, trained_file, mean_file, label_file);

    string file = argv[5];

    std::cout << "---------- Prediction for "
        << file << " ----------" << std::endl;

    // 读取待分类图像
    cv::Mat img = cv::imread(file, -1);

    // 这里开始用图片列表，并且显示出来 
    CHECK(!img.empty()) << "Unable to decode image " << file;

    // 分类
    std::vector<Prediction> predictions = classifier.Classify(img);

    // 打印前N 个预测值
    /* Print the top N predictions. */
    for (size_t i = 0; i < predictions.size(); ++i) 
    {
        Prediction p = predictions[i];
        std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
              << p.first << "\"" << std::endl;
    }
}
#else
int main(int argc, char** argv) 
{
    LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
