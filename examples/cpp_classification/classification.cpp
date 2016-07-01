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

/* pair(��ǩ�����Ŷ�)  Ԥ��ֵ */
/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

// ���࣬Ĭ�Ϸ���ǰ5��Ԥ��ֵ[(��ǩ�����Ŷ�),... ] ����
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

    shared_ptr<Net<float> > net_; // caffe�����������
    cv::Size input_geometry_; // �������ݵļ���ά��,��͸� 
    int num_channels_; // ͨ���� 
    cv::Mat mean_; // ��ֵͼ��
    std::vector<string> labels_; // Ŀ���ǩ����
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
    // �����������˽ṹ 
    net_.reset(new Net<float>(model_file, TEST));

    // ��������Ȩ�� 
    net_->CopyTrainedLayersFrom(trained_file);

    // ����glog�ļ�� 
    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";

    // ��� 
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

    // �����ģ�� Blob
    Blob<float>* input_layer = net_->input_blobs()[0];

    // ͨ����
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

    // ���ؾ�ֵ�ļ�
    /* Load the binaryproto mean file. */
    SetMean(mean_file);

    // ���ط����ǩ�ļ�
    /* Load labels. */
    std::ifstream labels(label_file.c_str());
    CHECK(labels) << "Unable to open labels file " << label_file;
    string line;
    
    while (std::getline(labels, line))
    labels_.push_back(string(line));

    Blob<float>* output_layer = net_->output_blobs()[0];

    // ���labels_�ĳ�����������ά���Ƿ�һ�� 
    CHECK_EQ(labels_.size(), output_layer->channels())
    << "Number of labels is different from the output layer dimension.";
}

// ������������������������ 
static bool PairCompare(const std::pair<float, int>& lhs,
                const std::pair<float, int>& rhs) 
{
    return lhs.first > rhs.first;
}

// ��������v[] ���ֵ��ǰ N ���������
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

// ���ಢ��������ǰ N ��Ԥ��
/* Return the top N predictions. */
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N) 
{
    std::vector<float> output = Predict(img);

    N = std::min<int>(labels_.size(), N);

    // ȡǰN��Ԥ���� 
    std::vector<int> maxN = Argmax(output, N);
    std::vector<Prediction> predictions;
    
    for (int i = 0; i < N; ++i) 
    {
        int idx = maxN[i];

        // ��� [(��ǩ�����Ŷ�),...]Ԥ��ֵ����
        predictions.push_back(std::make_pair(labels_[idx], output[idx]));
    }

    return predictions;
}

// �Ӷ����Ƶ�bin�ļ��ж�ȡ��ֵ�������õ�blob_��
/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file) 
{
    // ����google/protobuf?? ,���ڼ�����������ݽӿڣ���ʱ������ϸ�˽���Ӧ�÷��� 
    BlobProto blob_proto;
    // ���������ʵ���˴Ӷ������ļ��ж�ȡ���ݵ�blob_proto��,�²⺯�����Ե�3�����google/protobufģ�� 
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

    /* Convert from BlobProto to Blob<float> */
    Blob<float> mean_blob;

    // ����Blob��ĳ�Ա����FromRroto��BlobProto�м������� 
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), num_channels_)
    << "Number of channels of mean file doesn't match input layer.";

    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector<cv::Mat> channels;

    // �ÿɶ�д�ķ�ʽȡ��ָ�� 
    float* data = mean_blob.mutable_cpu_data();

    // �Ѿ�ֵ�ϵĸ���ͨ���ĸ��Ƶ� vector<Mat> channels����channels[0]�ж�Ӧ��ֵ�е�ͨ��0�� 
    // ��������ԭ���� Blob������ݴ洢��ʽ��һά�ġ� 
    // ���������ǰ�һά�ȵ����� ת��ΪMat������ 
    for (int i = 0; i < num_channels_; ++i) 
    {
        /* Extract an individual channel. */
        cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
        channels.push_back(channel);
        data += mean_blob.height() * mean_blob.width();
    }

    /* Merge the separate channels into a single image. */
    cv::Mat mean;

    // �ϲ��ֿ���ͨ��Ϊһ��ͼ�񣬼���channels������Mat�ϲ�Ϊһ��Mat. 
    cv::merge(channels, mean);

    // ����ÿ������������ͨ���ϵ�ƽ��ֵ��������channel_mean�� 
    /* Compute the global mean pixel value and create a mean image
    * filled with this value. */
    cv::Scalar channel_mean = cv::mean(mean);

    // ��ֵ�� ����ĳ�Ա����mean_ 
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

// ��ͼƬ����Ԥ�� 
std::vector<float> Classifier::Predict(const cv::Mat& img) 
{
    // �õ�net�����������ָ�� 
    Blob<float>* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, num_channels_,
                   input_geometry_.height, input_geometry_.width);

    /* Forward dimension change to all layers. */
    net_->Reshape();

    std::vector<cv::Mat> input_channels;

    // ��������
    // ��net_->input_blobs()[0]�ĵ�ַ��input_channels 
    WrapInputLayer(&input_channels);

    // ����Ԥ����
    // ��ͼƬ��ַ��input_channels 
    Preprocess(img, &input_channels);

    // ���в�ǰ����� 
    net_->Forward();

    // ��net����������ݸ��Ƶ�vector<float>���͵ı����У������� 
    /* Copy the output layer to a std::vector */
    Blob<float>* output_layer = net_->output_blobs()[0];
    const float* begin = output_layer->cpu_data();
    const float* end = begin + output_layer->channels();

    return std::vector<float>(begin, end);
}

// ��net_�����ݽӿ���input_channels �Խ� 
/* ��������в�ͬ�ĵ������ cv:Mat ����
  ��ÿ��ͨ��һ�������������Ǳ���һ�� memcpy�Ĳ���������
   ������Ҫ����cudaMemcpy2D �����Ԥ����
   ������ֱ��д�벻ͬͨ��������㡣
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

    // ȡ��Blob������ݣ����ں������ֶ�������޸�(����Preprocess�У���ͼƬ��ֵ����input_layer��) 
    float* input_data = input_layer->mutable_cpu_data();

    for (int i = 0; i < input_layer->channels(); ++i) 
    {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}

// ��imgΪ���룬��net_��forword���������ֵ�� 
void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) 
{
    // ��֤����ͼƬ��channels������channelsһ��
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;

    // ͨ�����ݸ������ý���ת��
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
    
    // ��֤��Сһ�� 
    cv::Mat sample_resized;
    
    if (sample.size() != input_geometry_)
    {
        cv::resize(sample, sample_resized, input_geometry_);
    }
    else
    {
        sample_resized = sample;
    }
    
    // ��֤��������һ��Ϊ float 
    cv::Mat sample_float;
    
    if (num_channels_ == 3)
    {
        sample_resized.convertTo(sample_float, CV_32FC3);
    }
    else
    {
        sample_resized.convertTo(sample_float, CV_32FC1);
    }
    
    // ��ȥ��ֵ�õ�sample_normalized 
    cv::Mat sample_normalized;
    cv::subtract(sample_float, mean_, sample_normalized);

    // ������BGRֱ��д����������input_channels
    // �� sample_normalized ���� input_channels�У�������net_->input_blob�С�
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
    
    // glog ���ں�����glog ����һ������־�Ŀ� 
    ::google::InitGoogleLogging(argv[0]);

    string model_file   = argv[1]; // ����ṹ�ļ���
    string trained_file = argv[2]; // ѵ��Ȩֵ�ļ���
    string mean_file    = argv[3]; // ��ֵ�ļ���
    string label_file   = argv[4]; // �����ļ���

    // ����������
    Classifier classifier(model_file, trained_file, mean_file, label_file);

    string file = argv[5];

    std::cout << "---------- Prediction for "
        << file << " ----------" << std::endl;

    // ��ȡ������ͼ��
    cv::Mat img = cv::imread(file, -1);

    // ���￪ʼ��ͼƬ�б�������ʾ���� 
    CHECK(!img.empty()) << "Unable to decode image " << file;

    // ����
    std::vector<Prediction> predictions = classifier.Classify(img);

    // ��ӡǰN ��Ԥ��ֵ
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
