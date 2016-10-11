#ifdef WITH_PYTHON_LAYER
#include "boost/python.hpp"
namespace bp = boost::python;
#endif

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/signal_handler.h"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::Solver;
using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;
using caffe::vector;
using std::ostringstream;

DEFINE_string(gpu, "",
    "Optional; run in GPU mode on given device IDs separated by ','."
    "Use '-gpu all' to run on all available GPUs. The effective training "
    "batch size is multiplied by the number of devices.");
DEFINE_string(solver, "",
    "The solver definition protocol buffer text file.");
DEFINE_string(model, "",
    "The model definition protocol buffer text file..");
DEFINE_string(snapshot, "",
    "Optional; the snapshot solver state to resume training.");
DEFINE_string(weights, "",
    "Optional; the pretrained weights to initialize finetuning, "
    "separated by ','. Cannot be set simultaneously with snapshot.");
DEFINE_int32(iterations, 50,
    "The number of iterations to run.");
DEFINE_string(sigint_effect, "stop",
             "Optional; action to take when a SIGINT signal is received: "
              "snapshot, stop or none.");
DEFINE_string(sighup_effect, "snapshot",
             "Optional; action to take when a SIGHUP signal is received: "
             "snapshot, stop or none.");
/*
（3）g_brew_map实现过程，首先通过typedef定义函数指针
typedef int (*BrewFunction)();
这个是用typedef定义函数指针方法。这个程序定义一个BrewFunction函数指针类型，
在caffe.cpp 中 BrewFunction 作为GetBrewFunction()函数的返回类型，可以是train()，
test()，device_query()，time() 这四个函数指针的其中一个。在train()，test()，中
可以调用solver类的函数，从而进入到net，进入到每一层，运行整个caffe程序。

在main函数中出现了GetBrewFunction函数，即在标准指令下，main函数将执行GetBrewFunction函数。首先看看caffe.cpp中一些重要代码：
// 这里定义函数指针类型BrewFunction
typedef int (*BrewFunction)();
// c++标准map容器，caffe执行的action name与对应函数的映射，容器类型名为BrewMap
typedef std::map BrewMap;
// 声明map容器变量g_brew_map
BrewMap g_brew_map;

// 宏定义，比如RegisterBrewFunction（train）时，相当于在容器g_brew_map中注册了train函数的函数指针和其对应的名字“train”，对于#和##的用法见下文。
#define RegisterBrewFunction(func) \
namespace { \
class __Registerer_##func { \
 public: \
  __Registerer_##func() { \
    g_brew_map[#func] = &func; \
  } \
}; \
__Registerer_##func g_registerer_##func; \
}
C++中#和##用法:在C/C++的宏中，”#”的功能是将其后面的宏参数进行字符串化操作(Stringfication)，简单说就是在对它所引用的宏变量通过替换后在其左右各加上一个双引号。”##”被称为连接符(concatenator)，用来将两个子串Token连接为一个Token。注意这里连接的对象是Token就行，而不一定是宏的变量。
凡是宏定义里有用’#’或’##’的地方宏参数是不会再展开。若要使’#’和’##’的宏参数被展开，可以加多一层中间转换宏。

在caffe.cpp中定义了一些BrewFunction类的函数，通过RegisterBrewFunction（function）注册进容器g_brew_map：
int device_query() ：用来查询GPU信息
int train()：训练神经网络
int test() ：测试神经网络
int time()：测试model执行时间

GetBrewFunction函数通过caffe命令后第一个参数在g_brew_map容器中查找对应函数指针并返回。代码如下：
static BrewFunction GetBrewFunction(const caffe::string& name) {
  if (g_brew_map.count(name)) {
    return g_brew_map[name];
  } else {
    LOG(ERROR) << "Available caffe actions:";
    for (BrewMap::iterator it = g_brew_map.begin();
         it != g_brew_map.end(); ++it) {
      LOG(ERROR) << "\t" << it->first;
    }
    LOG(FATAL) << "Unknown action: " << name;
    return NULL;  // not reachable, just to suppress old compiler warnings.
  }
}
*/
// A simple registry for caffe commands.
typedef int (*BrewFunction)();

/*
（4）g_brew_map定义
typedef std::map<caffe::string, BrewFunction> BrewMap;
因为输入参数可能为train,test，device_query，time，所以定义一个容器类型，
*/
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;

// （5） g_brew_map 初始化
// 这个作用和#define RegisterBrewFunction(func) g_brew_map[#func]=&func;
// 这个宏定义功能类似，其中，func可以为：train，test，device_query，time。
#define RegisterBrewFunction(func) \
namespace { \
class __Registerer_##func { \
 public: /* NOLINT */ \
  __Registerer_##func() { \
    g_brew_map[#func] = &func; \
  } \
}; \
__Registerer_##func g_registerer_##func; \
}

// （2）GetBrewFunction()函数定义如下,其返回BrewFunction函数指针。
static BrewFunction GetBrewFunction(const caffe::string& name) 
{
    // 判断输入的是不是g_brew_map中train，test，device_query，time中一个，
    if (g_brew_map.count(name)) 
    {
        // 如果是的话，就调用相应的train(),test()，device_query()，time()
        return g_brew_map[name];
    } 
    else 
    {
        LOG(ERROR) << "Available caffe actions:";
        
        for (BrewMap::iterator it = g_brew_map.begin();
             it != g_brew_map.end(); ++it) 
        {
          LOG(ERROR) << "\t" << it->first;
        }
             
        LOG(FATAL) << "Unknown action: " << name;
        return NULL;  // not reachable, just to suppress old compiler warnings.
    }
}

// caffe中定义了train()，test()，device_query()，time()四种方式.如果需要，咱们
// 可以增加其他的方式，然后通过RegisterBrewFunction() 函数注册一下即可。    

// Parse GPU ids or use all available devices
static void get_gpus(vector<int>* gpus) {
  if (FLAGS_gpu == "all") {
    int count = 0;
#ifndef CPU_ONLY
    CUDA_CHECK(cudaGetDeviceCount(&count));
#else
    NO_GPU;
#endif
    for (int i = 0; i < count; ++i) {
      gpus->push_back(i);
    }
  } else if (FLAGS_gpu.size()) {
    vector<string> strings;
    boost::split(strings, FLAGS_gpu, boost::is_any_of(","));
    for (int i = 0; i < strings.size(); ++i) {
      gpus->push_back(boost::lexical_cast<int>(strings[i]));
    }
  } else {
    CHECK_EQ(gpus->size(), 0);
  }
}

// caffe commands to call by
//     caffe <command> <args>
//
// To add a command, define a function "int command()" and register it with
// RegisterBrewFunction(action);

// Device Query: show diagnostic information for a GPU device.
int device_query() {
  LOG(INFO) << "Querying GPUs " << FLAGS_gpu;
  vector<int> gpus;
  get_gpus(&gpus);
  for (int i = 0; i < gpus.size(); ++i) {
    caffe::Caffe::SetDevice(gpus[i]);
    caffe::Caffe::DeviceQuery();
  }
  return 0;
}
RegisterBrewFunction(device_query);

// Load the weights from the specified caffemodel(s) into the train and
// test nets.
void CopyLayers(caffe::Solver<float>* solver, const std::string& model_list) 
{
    std::vector<std::string> model_names;
    boost::split(model_names, model_list, boost::is_any_of(",") );

    for (int i = 0; i < model_names.size(); ++i) 
    {
        LOG(INFO) << "Finetuning from " << model_names[i];
        solver->net()->CopyTrainedLayersFrom(model_names[i]);

        for (int j = 0; j < solver->test_nets().size(); ++j) 
        {
            solver->test_nets()[j]->CopyTrainedLayersFrom(model_names[i]);
        }
    }
}

// Translate the signal effect the user specified on the command-line to the
// corresponding enumeration.
caffe::SolverAction::Enum GetRequestedAction(
    const std::string& flag_value) {
  if (flag_value == "stop") {
    return caffe::SolverAction::STOP;
  }
  if (flag_value == "snapshot") {
    return caffe::SolverAction::SNAPSHOT;
  }
  if (flag_value == "none") {
    return caffe::SolverAction::NONE;
  }
  LOG(FATAL) << "Invalid signal effect \""<< flag_value << "\" was specified";
}

// Train / Finetune a model.
int train() 
{
    // 检查输入参数solver,snapshot和weight。其中solver为solver的ptototxt文件  
    // snapshot为训练时产生的快照，以便在训练中断后，不至于从头开始训练  
    // weights为一个已有的训练好的网络，如果指定了weights，则训练的时候会用指定  
    // 的weights初始化网络参数，然后再训练，主要用于对网络进行finetune  
    // 注意:snapshot和weights不能同时使用  
    CHECK_GT(FLAGS_solver.size(), 0) << "Need a solver definition to train.";
    CHECK(!FLAGS_snapshot.size() || !FLAGS_weights.size())
    << "Give a snapshot to resume training or weights to finetune "
    "but not both.";

    // 从指定的solver的prototxt文件中读取SolverParameter 
    // 实例化SolverParameter类，该类保存solver参数和相应的方法（SoverParameter
    // 是由google protobuffer编译过来的类，具体声明可以见代码文件build/src/caffe/proto/caffe.pb.h）；
    caffe::SolverParameter solver_param;

    // 将-solver指定solver.prototxt文件内容解析到solver_param中，该函数声明在
    // include/caffe/util/upgrade_proto.hpp中，实现在src/caffe/util/upgrade_proto.cpp中；
    caffe::ReadSolverParamsFromTextFileOrDie(FLAGS_solver, &solver_param);

    // 根据命令参数-gpu或者solver.prototxt提供的信息设置GPU；
    // If the gpus flag is not provided, allow the mode and device to be set
    // in the solver prototxt.
    if (FLAGS_gpu.size() == 0
     && solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU) 
    {
        if (solver_param.has_device_id()) 
        {
            FLAGS_gpu = "" +
                boost::lexical_cast<string>(solver_param.device_id());
        } 
        else 
        {
            // boost::lexical_cast(0)是将数值0转换为字符串'“0”；
            // Set default GPU if unspecified
            FLAGS_gpu = "" + boost::lexical_cast<string>(0);
        }
    }

    // 多GPU下，将GPU编号存入vector容器中（get_gpus()函数通过FLAGS_gpu获取）；
    vector<int> gpus;
    get_gpus(&gpus);
    
    if (gpus.size() == 0) 
    {
        LOG(INFO) << "Use CPU.";
        Caffe::set_mode(Caffe::CPU);
    } 
    else 
    {
        ostringstream s;
        
        for (int i = 0; i < gpus.size(); ++i) 
        {
            s << (i ? ", " : "") << gpus[i];
        }
        
        LOG(INFO) << "Using GPUs " << s.str();
        
#ifndef CPU_ONLY
        cudaDeviceProp device_prop;

        for (int i = 0; i < gpus.size(); ++i) 
        {
            cudaGetDeviceProperties(&device_prop, gpus[i]);
            LOG(INFO) << "GPU " << gpus[i] << ": " << device_prop.name;
        }
#endif

        solver_param.set_device_id(gpus[0]);
        Caffe::SetDevice(gpus[0]);
        Caffe::set_mode(Caffe::GPU);
        Caffe::set_solver_count(gpus.size());
    }

    // 处理snapshot, stop or none信号，其声明在include/caffe/util/signal_Handler.h中；
    // GetRequestedAction在caffe.cpp中，将‘stop’，‘snapshot’，‘none’转换为标准信号，即解析；
    caffe::SignalHandler signal_handler(
     GetRequestedAction(FLAGS_sigint_effect),
     GetRequestedAction(FLAGS_sighup_effect));

    /*
    solver作用：（指定优化方法）
    1.可以逐步对网络寻优，创建训练得到的网络，并对测试网络进行评价；
    2.通过调用forward和backward来对网络参数进行迭代寻优；
    3.周期性更新网络；
    4.记录网络训练中间过程，寻优过程中记录状态 
    */

    // 声明boost库中智能指针solver，指向caffe::Solver对象，该对象由CreateSolver创建
    // 用读取的SolverParameter创建Solver  
    shared_ptr<caffe::Solver<float> >
    solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));

    // Solver对象中方法的使用
    solver->SetActionFunction(signal_handler.GetActionFunction());

    // createSolver()这里应该是选择了默认的sgd
    // boost::shared_ptr 指针
    // 初始化solver
    // 利用snapshot restore网络或利用weights初始化网络的参数
    // 从snapshot或caffemodel中恢复train；
    if (FLAGS_snapshot.size()) 
    {
        LOG(INFO) << "Resuming from " << FLAGS_snapshot;
        solver->Restore(FLAGS_snapshot.c_str());
    } 
    else if (FLAGS_weights.size()) 
    {
        CopyLayers(solver.get(), FLAGS_weights);
    }

    // 进行训练  
    if (gpus.size() > 1) 
    {
        // 这里是对于多GPU下的处理
        caffe::P2PSync<float> sync(solver, NULL, solver->param());
        sync.Run(gpus);
    } 
    else 
    {
        LOG(INFO) << "Starting Optimization";

        // 初始化完成，开始优化网络（核心，重要）；
        solver->Solve();
    }
    
    LOG(INFO) << "Optimization Done.";
    return 0;
}
RegisterBrewFunction(train);


// Test: score a model.
int test() 
{
    CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
    CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";

    // Set device id and mode
    vector<int> gpus;
    get_gpus(&gpus);
    
    if (gpus.size() != 0) 
    {
        LOG(INFO) << "Use GPU with device ID " << gpus[0];
#ifndef CPU_ONLY
        cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, gpus[0]);
        LOG(INFO) << "GPU device name: " << device_prop.name;
#endif
        Caffe::SetDevice(gpus[0]);
        Caffe::set_mode(Caffe::GPU);
    } 
    else 
    {
        LOG(INFO) << "Use CPU.";
        Caffe::set_mode(Caffe::CPU);
    }
    
    // Instantiate the caffe net.
    Net<float> caffe_net(FLAGS_model, caffe::TEST);
    caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
    LOG(INFO) << "Running for " << FLAGS_iterations << " iterations.";

    vector<int> test_score_output_id;
    vector<float> test_score;
    float loss = 0;
    
    for (int i = 0; i < FLAGS_iterations; ++i) 
    {
        float iter_loss;
        const vector<Blob<float>*>& result = caffe_net.Forward(&iter_loss);
        loss += iter_loss;
        int idx = 0;
        
        for (int j = 0; j < result.size(); ++j) 
        {
            const float* result_vec = result[j]->cpu_data();
            
            for (int k = 0; k < result[j]->count(); ++k, ++idx) 
            {
           
                const float score = result_vec[k];
                
                if (i == 0) 
                {
                    test_score.push_back(score);
                    test_score_output_id.push_back(j);
                } 
                else 
                {
                    test_score[idx] += score;
                }
                
                const std::string& output_name = caffe_net.blob_names()[
                    caffe_net.output_blob_indices()[j]];
                LOG(INFO) << "Batch " << i << ", " << output_name << " = " << score;
            }
        }
    }
    
    loss /= FLAGS_iterations;
    LOG(INFO) << "Loss: " << loss;
    
    for (int i = 0; i < test_score.size(); ++i) 
    {
        const std::string& output_name = caffe_net.blob_names()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
        
        const float loss_weight = caffe_net.blob_loss_weights()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
        
        std::ostringstream loss_msg_stream;
        const float mean_score = test_score[i] / FLAGS_iterations;
        
        if (loss_weight) 
        {
            loss_msg_stream << " (* " << loss_weight
                          << " = " << loss_weight * mean_score << " loss)";
        }
        
        LOG(INFO) << output_name << " = " << mean_score << loss_msg_stream.str();
    }

    return 0;
}
RegisterBrewFunction(test);


// Time: benchmark the execution time of a model.
int time() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to time.";

  // Set device id and mode
  vector<int> gpus;
  get_gpus(&gpus);
  if (gpus.size() != 0) {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TRAIN);

  // Do a clean forward and backward pass, so that memory allocation are done
  // and future iterations will be more stable.
  LOG(INFO) << "Performing Forward";
  // Note that for the speed benchmark, we will assume that the network does
  // not take any input blobs.
  float initial_loss;
  caffe_net.Forward(&initial_loss);
  LOG(INFO) << "Initial loss: " << initial_loss;
  LOG(INFO) << "Performing Backward";
  caffe_net.Backward();

  const vector<shared_ptr<Layer<float> > >& layers = caffe_net.layers();
  const vector<vector<Blob<float>*> >& bottom_vecs = caffe_net.bottom_vecs();
  const vector<vector<Blob<float>*> >& top_vecs = caffe_net.top_vecs();
  const vector<vector<bool> >& bottom_need_backward =
      caffe_net.bottom_need_backward();
  LOG(INFO) << "*** Benchmark begins ***";
  LOG(INFO) << "Testing for " << FLAGS_iterations << " iterations.";
  Timer total_timer;
  total_timer.Start();
  Timer forward_timer;
  Timer backward_timer;
  Timer timer;
  std::vector<double> forward_time_per_layer(layers.size(), 0.0);
  std::vector<double> backward_time_per_layer(layers.size(), 0.0);
  double forward_time = 0.0;
  double backward_time = 0.0;
  for (int j = 0; j < FLAGS_iterations; ++j) {
    Timer iter_timer;
    iter_timer.Start();
    forward_timer.Start();
    for (int i = 0; i < layers.size(); ++i) {
      timer.Start();
      layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
      forward_time_per_layer[i] += timer.MicroSeconds();
    }
    forward_time += forward_timer.MicroSeconds();
    backward_timer.Start();
    for (int i = layers.size() - 1; i >= 0; --i) {
      timer.Start();
      layers[i]->Backward(top_vecs[i], bottom_need_backward[i],
                          bottom_vecs[i]);
      backward_time_per_layer[i] += timer.MicroSeconds();
    }
    backward_time += backward_timer.MicroSeconds();
    LOG(INFO) << "Iteration: " << j + 1 << " forward-backward time: "
      << iter_timer.MilliSeconds() << " ms.";
  }
  LOG(INFO) << "Average time per layer: ";
  for (int i = 0; i < layers.size(); ++i) {
    const caffe::string& layername = layers[i]->layer_param().name();
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername <<
      "\tforward: " << forward_time_per_layer[i] / 1000 /
      FLAGS_iterations << " ms.";
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername  <<
      "\tbackward: " << backward_time_per_layer[i] / 1000 /
      FLAGS_iterations << " ms.";
  }
  total_timer.Stop();
  LOG(INFO) << "Average Forward pass: " << forward_time / 1000 /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Average Backward pass: " << backward_time / 1000 /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Average Forward-Backward: " << total_timer.MilliSeconds() /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Total Time: " << total_timer.MilliSeconds() << " ms.";
  LOG(INFO) << "*** Benchmark ends ***";
  return 0;
}
RegisterBrewFunction(time);

/*
总结一下程序运行流程：
main（）函数--->>GetBrewFunction函数--->>train函数--->>Solve()

caffe-master/Tools文件夹下提供了caffe框架的主要工具（经编译后为可执行文件，在build/tools/下）。
tools/caffe.cpp是caffe程序的入口（即main函数），一条标准的训练指令为：
./build/tools/caffe train --solver=models/bvlc_reference_caffenet/solver.prototxt
首先是caffe指令，可执行指令，train为caffe指令第一条参数，然后是指定solver文件。
我们对照着该标准指令一步一步来“解析”，caffe.cpp中main函数代码如下：
*/
int main(int argc, char** argv) 
{
    // gflags库，具体说明紧接代码（未找到其定义，估计在gflags库文件中定义）
    // Print output to stderr (while still logging).
    FLAGS_alsologtostderr = 1;
    
    // Set version
    gflags::SetVersionString(AS_STRING(CAFFE_VERSION));

    // gflags库中为main函数设置usage信息：extern void SetUsageMessage(const std::string& usage);
    // Usage message.
    gflags::SetUsageMessage("command line brew\n"
    "usage: caffe <command> <args>\n\n"
    "commands:\n"
    "  train           train or finetune a model\n"
    "  test            score a model\n"
    "  device_query    show GPU diagnostic information\n"
    "  time            benchmark model execution time");

    // include/caffe/commom.hpp中声明的函数：Currently it initializes google flags and google logging.即初始化FLAGS.
    // Run tool or show usage.
    caffe::GlobalInit(&argc, &argv);

    // 判断参数，参数为2，继续执行action函数，否则输出usage信息。
    // （1）main()函数中，输入的train，test，device_query，time。 通过下面两行进入程序。
    if (argc == 2) 
    {
#ifdef WITH_PYTHON_LAYER
        try 
        {
#endif
            // GetBrewFunction函数返回函数指针，对于上面标准指令，则返回train函数指针
            // 最后是执行相应的函数，如执行train函数，执行成功则返回0，main函数返回0.（caffe执行完毕）
            return GetBrewFunction(caffe::string(argv[1]))();
#ifdef WITH_PYTHON_LAYER
        } 
        catch (bp::error_already_set) 
        {
            PyErr_Print();
            return 1;
        }
#endif
    } 
    else 
    {
        // glags中为main函数提供usage信息：
        // extern void ShowUsageWithFlags(const char *argv0);  // what --help does
        // extern void ShowUsageWithFlagsRestrict(const char *argv0, const char *restrict);
        // 其信息中会有“tools/caffe.cpp”中FLAG信息，如：-gpu,-weights,-solver,-snapshot,-model...
        gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/caffe");
    }
}

/*
gflags是google的一个开源的处理命令行参数的库。在使用命令行参数的文件文件中（源文
件或头文件），首先使用一下定义语句进行变量的定义。DEFINE_int32，DEFINE_int64，
DEFINE_bool，DEFINE_double，DEFINE_string等，语法为：DEFINE_int32(name, 
default_value, "description")。接着你就可以使用FLAGS_name变量了，这些变量的值则
是由命令行参数传递，无则为默认值，在其他代码文件中若想用该命令参数，可以用
DECLARE_int32(name)声明（name为int32类型，也可以使用其他支持的类型）。在
caffe.cpp中有很多FLAGS_name定义，如DEFINE_string(gpu,"","some description"），
则命令行后-gpu 0，表示FLAGS_gpu=0，默认值为空。
*/
