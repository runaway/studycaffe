#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "hdf5.h"

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/parallel.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/insert_splits.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/upgrade_proto.hpp"

#include "caffe/test/test_caffe_main.hpp"
// 主要定义了一个模板类net
/*
Net类是Solve类的一个成员，在net.cpp中定义了对Net的所有操作，其中包括：
Init
GetLearningRateAndWeightDecay
ForwardPrefilled
Backward
ShareTrainedLayersWith
CopyTrainedLayersFrom
ToProto
Update
has_blob
blob_by_name
has_layer
layer_by_name
*/

namespace caffe 
{
/*
功能：调用Init函数初始化网络 
输入：NetParameter& param 
输出：无
*/
template <typename Dtype>
Net<Dtype>::Net(const NetParameter& param, const Net* root_net)
    : root_net_(root_net) 
{
    Init(param);
}

/*
功能：调用Init函数初始化网络 
输入：string& param_file 
输出：无
*/
template <typename Dtype>
Net<Dtype>::Net(const string& param_file, Phase phase, const Net* root_net)
    : root_net_(root_net) 
{
    NetParameter param;
    ReadNetParamsFromTextFileOrDie(param_file, &param);
    param.mutable_state()->set_phase(phase);
    Init(param);
}

/*
功能：初始化网络
输入：NetParameter& in_param
输出：无
步骤：
<1> 调用InsertSplits()函数从in_param读入新网络到param
<2> 定义name_，blob_name_to_idx，available_blobs，num_layers
<3> param.input_size()返回输入层blob的个数;
    param.input(i)表示第i个blob的名字;
    param.layers_size()返回网络的层数。
<4> 对每一个输入层的blob：
    产生一块和当前blob一样大的空间 e.g. imput_dim=[12 55 66 39 20 24 48 64]表示
    第一个blob的四个维数为 12 55 66 39，第二个为 20 24 48 64 接着blob_pointer指
    向这块空间
    blob_pointer压到blobs_中 vector<shared_ptr<Blob<Dtype>>> blobs_
    blob_name压到blob_names_中 vector<string> blob_names_
    param.force_backward()压到blob_need_backward_中vector<bool> blob_need_backward_
    i 压到 net_input_blob_indices_中 net_input_blob_indices_ -> vector
    blob_pointer.get() 压到 net_input_blobs_中
    注意与blobs_的区别
    vector<shared_ptr<Blob<Dtype>>> blobs_
    vector<Blob<Dtype>*> net_input_blobs_
    shared_ptr类型的参数调用.get()则得到Blob*类型
    map<string, int> blob_name_to_idx
    初始化为输入层的每个blob的名字 set<string> available_blobs
    计算所需内存 memory_used += blob_pointer->count()

<5> 存每一层的输入blob指针 vector<vector<Blob<Dtype>*> > bottom_vecs_
    存每一层输入(bottom)的id vector<vector<int> > bottom_id_vecs_
    存每一层输出(top)的blob vector<vector<Blob<Dtype>*> > top_vecs_
    用网络的层数param.layers_size()去初始化上面四个变量
    vector<vector<int> > top_id_vecs_
<6> 对第i层（很大的一个for循环）：
    param.layers(i)返回的是关于第当前层的参数：
    layer_param = param.layers(i)
    把当前层的参数转换为shared_ptr<Layer<Dtype>>，并压入到layers_中
    把当前层的名字压入到layer_names_：vector<string> layer_names_
    判断当前层是否需要反馈 need_backward = param.force_backward()

    下面开始产生当前层：分为处理bottom的blob和top的blob两个步骤
    对第j个bottom的blob：
        layer_param.bottom_size()存的是当前层的输入blob数量
        layer_param.bottom(j)存的是第j个输入blob的名字
        读取当前blob的id，其中blob_name_to_idx在输入层初始化过了
        blob_name_to_idx[blob_name] = i
        输出当前blob的名字
        存入第j个输入blob的指针bottom_vecs_[i].push_back(blobs_[blob_id].get())
        存入第j个输入blob的id bottom_id_vecs_[i].push_back(blob_id)
        更新need_backward
        从available_blobs中删除第j个blob的名字

    对第j个top的blob：
        layer_param.top_size()存的是当前层的输出blob数量
        layer_param.top(j)存的是第j个输出blob的名字
        判断是否进行同址计算
        输出当前blob的名字
        定义一块新的blob空间，用blob_pointer指向这块空间
        把这个指针存入到blobs_中
        把blob_name、force_backward、idx存入对应的容器中
        向available_blobs插入当前blob的名字
        top_vecs_[i]对于第i层，插入当前blob的指针
        top_id_vecs_[i]对于第i层，插入当前blob的id
    输出当前层位于top的blob的信息
    计算所需内存
    判断当前层i是否需要backward

<7> 所有名字在available_blobs中的blob为当前层的输出blob，存入net_output_blobs_中
<8> 建立每个blob的name和index的对应关系map：blob_names_index_
<9> 建立每个层的name和index的对应关系map：layer_names_index_
<10> 调用GetLearningRateAndWeightDecay函数


模型初始化使用Net::Init().这个初始化主要做两件事：创建blobs和layers搭建整个有向
无环图（DAG），调用layers的Setup()函数。它也做一些统计工作，例如校验整个网络架
构的正确性。

注意：网络的构建是设备无关的。构建之后，网络是运行在CPU或GPU上是通过一个单独的
定义实现的Caffe::mode()，设置Caffe::set_mode()。

模型是在纯文本protocol buffer模式.prototxt中定义的，学习好的模型被序列化为binary
protocol buffer，存储在 .caffemodel文件中。

caffe使用Google Protocol Buffer出于以下几个优点：

序列化时最小化binary string的size，有效序列化，文本格式兼容binary version，在多
种语言中都有接口实现，例如C++和Python。这些优点使得在caffe建模灵活可拓展。
*/
template <typename Dtype>
void Net<Dtype>::Init(const NetParameter& in_param) 
{
    CHECK(Caffe::root_solver() || root_net_)
      << "root_net_ needs to be set for all non-root solvers";
    
    // Set phase from the state.
    phase_ = in_param.state().phase();
    
    // Filter layers based on their include/exclude rules and
    // the current NetState.
    NetParameter filtered_param;
    FilterNet(in_param, &filtered_param);
    LOG_IF(INFO, Caffe::root_solver())
      << "Initializing net from parameters: " << std::endl
      << filtered_param.DebugString();
    
    // Create a copy of filtered_param with splits added where necessary.
    NetParameter param;

    // <1> 调用InsertSplits()函数从in_param读入新网络到param 
    InsertSplits(filtered_param, &param);

    // <2> 定义name_，blob_name_to_idx，available_blobs，num_layers 
    // Basically, build all the layers and set up their connections.
    name_ = param.name();

    // 7. map<string, int> blob_name_to_idx
    // blob_name_to_idx是一个map,其关键字是不重复的  
    map<string, int> blob_name_to_idx;

    // 8. 初始化为输入层的每个blob的名字 set<string> available_blobs
    //available_blobs是一个set,其关键字是不重复的  
    set<string> available_blobs;
    memory_used_ = 0;

    // <3> param.input_size()返回输入层blob的个数; 
    // param.input(i)表示第i个blob的名字; 
    // param.layers_size()返回网络的层数。 
    // For each layer, set up its input and output
    bottom_vecs_.resize(param.layer_size());
    top_vecs_.resize(param.layer_size());
    bottom_id_vecs_.resize(param.layer_size());
    param_id_vecs_.resize(param.layer_size());
    top_id_vecs_.resize(param.layer_size());
    bottom_need_backward_.resize(param.layer_size());

    // 用网络的层数param.layers_size()去初始化上面四个变量 

    // <4> 对每一个输入层的blob：
    // <6> 对第i层（很大的一个for循环）：
    for (int layer_id = 0; layer_id < param.layer_size(); ++layer_id) 
    {
        // For non-root solvers, whether this layer is shared from root_net_.
        bool share_from_root = !Caffe::root_solver()
        && root_net_->layers_[layer_id]->ShareInParallel();

        // 1. param.layers(i)返回的是关于第当前层的参数： 
        // Inherit phase from net if unset.
        if (!param.layer(layer_id).has_phase()) 
        {
            // 实参phase_是网络的phase,为模板类layer设置shape_属性      
            param.mutable_layer(layer_id)->set_phase(phase_);
        }

        // Setup layer.
        const LayerParameter& layer_param = param.layer(layer_id);

        // 检查LayerParameter类型propagate_down成员的个数是否达标 
        if (layer_param.propagate_down_size() > 0) 
        {
            // layer_param.bottom_size()存的是当前层的输入blob数量
            CHECK_EQ(layer_param.propagate_down_size(),
              layer_param.bottom_size())
              << "propagate_down param must be specified "
              << "either 0 or bottom_size times ";
        }

        // 2. 把当前层的参数转换为shared_ptr<Layer<Dtype>>，并压入到layers_中
        if (share_from_root) 
        {
            LOG(INFO) << "Sharing layer " << layer_param.name() << " from root net";
            layers_.push_back(root_net_->layers_[layer_id]);

            // 调用的是模板类Layer的SetShared方法  
            layers_[layer_id]->SetShared(true); 
        } 
        else 
        {
            // 注意这里的createlayer!
            // layer_factory.hpp 分析见下
            layers_.push_back(LayerRegistry<Dtype>::CreateLayer(layer_param));
        }

        // 3. 把当前层的名字压入到layer_names_：vector<string> layer_names_
        // 为layer_names_添加新元素  
        layer_names_.push_back(layer_param.name());
        LOG_IF(INFO, Caffe::root_solver())
        << "Creating Layer " << layer_param.name();
        bool need_backward = false;

        // 计算本层的输入和输出
        // Figure out this layer's input and output
        for (int bottom_id = 0; 
             bottom_id < layer_param.bottom_size();
             ++bottom_id) 
        {
            const int blob_id = AppendBottom(param, layer_id, bottom_id,
                                           &available_blobs, &blob_name_to_idx);

            // 4. 判断当前层是否需要反馈
            // 在遍历所有的bottom_id的过程中，只要有一次使得need_backward为真，
            // 则这个for循环结束后，need_backward也为真。也就是说该层前一层的
            // top blob中只要有一个blob在blob_need_backward_中为true，则
            // backward就为true，后面的layer_need_backward_也就push_back(true)  
            // If a blob needs backward, this layer should provide it.
            need_backward |= blob_need_backward_[blob_id];
        }

        // layer_param.top_size()存的是当前层的输出blob数量
        int num_top = layer_param.top_size();

        // 5. 下面开始产生当前层：分为处理bottom的blob和top的blob两个步骤 
        for (int top_id = 0; top_id < num_top; ++top_id) 
        {
            // 2. blob_pointer压到blobs_中 vector<shared_ptr<Blob<Dtype>>> blobs_
            // 3. blob_name压到blob_names_中 vector<string> blob_names_
            // 6. blob_pointer.get() 压到 net_input_blobs_中 
            // 注意与blobs_的区别 
            // 在AppendTop函数中，会为向量blob_need_backward_添加新元素  
            //vector<shared_ptr<Blob<Dtype>>> blobs_ 
            //vector<Blob<Dtype>*> net_input_blobs_ 
            //shared_ptr类型的参数调用.get()则得到Blob*类型
            AppendTop(param, layer_id, top_id, &available_blobs, &blob_name_to_idx);

            // Collect Input layer tops as Net inputs.
            if (layer_param.type() == "Input") 
            {
                const int blob_id = blobs_.size() - 1;

                // 5. i压到net_input_blob_indices_中 net_input_blob_indices_ -> vector
                net_input_blob_indices_.push_back(blob_id);
                net_input_blobs_.push_back(blobs_[blob_id].get());
            }
        }

        // If the layer specifies that AutoTopBlobs() -> true and the LayerParameter
        // specified fewer than the required number (as specified by
        // ExactNumTopBlobs() or MinTopBlobs()), allocate them here.
        Layer<Dtype>* layer = layers_[layer_id].get();
        
        if (layer->AutoTopBlobs()) 
        {
            const int needed_num_top =
              std::max(layer->MinTopBlobs(), layer->ExactNumTopBlobs());
            
            for (; num_top < needed_num_top; ++num_top) 
            {
                // Add "anonymous" top blobs -- do not modify available_blobs or
                // blob_name_to_idx as we don't want these blobs to be usable as input
                // to other layers.
                AppendTop(param, layer_id, num_top, NULL, NULL);
            }
        }
        
        // After this layer is connected, set it up.
        if (share_from_root) 
        {
            // Set up size of top blobs using root_net_
            const vector<Blob<Dtype>*>& base_top = root_net_->top_vecs_[layer_id];
            const vector<Blob<Dtype>*>& this_top = this->top_vecs_[layer_id];

            for (int top_id = 0; top_id < base_top.size(); ++top_id) 
            {
                this_top[top_id]->ReshapeLike(*base_top[top_id]);
                LOG(INFO) << "Created top blob " << top_id << " (shape: "
                    << this_top[top_id]->shape_string() <<  ") for shared layer "
                    << layer_param.name();
            }
        } 
        else 
        {
            // 注意这里的Setup!见底下关于layer.hpp的分析。
            // 调用模板类layer的SetUp方法，如果在网络的定义文件里没有设置
            // loss_weight，那么loss layer的LayerSetup函数里会设置loww_weght, 且默认值  
            layers_[layer_id]->SetUp(bottom_vecs_[layer_id], top_vecs_[layer_id]);
        }
        
        LOG_IF(INFO, Caffe::root_solver())
        << "Setting up " << layer_names_[layer_id];

        // 每次循环，都会更新向量blob_loss_weights  
        for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) 
        {
            if (blob_loss_weights_.size() <= top_id_vecs_[layer_id][top_id]) 
            {
                blob_loss_weights_.resize(top_id_vecs_[layer_id][top_id] + 1, Dtype(0));
            }

            // top_id_vecs_中存储的最基本元素是blob_id ——> 每一个新的blob都会
            // 赋予其一个blob_id，但是这个blob_id可能是会有重复的  
            // loss函数返回loss_weight ——> 在模板类的SetUp方法中会调用
            // SetLossWeights来设置其私有数据成员loss_,里面存储的其实是loss_weight  
            blob_loss_weights_[top_id_vecs_[layer_id][top_id]] = layer->loss(top_id);
            LOG_IF(INFO, Caffe::root_solver())
              << "Top shape: " << top_vecs_[layer_id][top_id]->shape_string();
            
            if (layer->loss(top_id)) 
            {
                LOG_IF(INFO, Caffe::root_solver())
                    << "    with loss weight " << layer->loss(top_id);
            }

            // 9. 计算所需内存 memory_used += blob_pointer->count()
            memory_used_ += top_vecs_[layer_id][top_id]->count();
        }
        
        LOG_IF(INFO, Caffe::root_solver())
        << "Memory required for data: " << memory_used_ * sizeof(Dtype);
        const int param_size = layer_param.param_size();
        const int num_param_blobs = layers_[layer_id]->blobs().size();

        // param_size是Layermeter类型对象layer_param中ParamSpec param成员的个数, num_param_blobs是一个Layer中learnable parameter blob的个数，param_size <= num_param_blobs  
        CHECK_LE(param_size, num_param_blobs)
        << "Too many params specified for layer " << layer_param.name();
        ParamSpec default_param_spec;
        
        for (int param_id = 0; param_id < num_param_blobs; ++param_id) 
        {
            const ParamSpec* param_spec = (param_id < param_size) ?
              &layer_param.param(param_id) : &default_param_spec;
            const bool param_need_backward = param_spec->lr_mult() != 0; // need backward 则为真。  

            // 由 param_need_backward 来决定need_backward是否为真(网络定义文件
            // 中的lr_mult很重要)，并且，只要有一次遍历使得need_backward为真，
            // 则这个for循环结束后，need_backward也为真  
            need_backward |= param_need_backward;

            // 设定一个Layer的parameter blob 是否需要计算diff backward->set_param_propagate_down
            // 是模板类Layer的方法。  
            layers_[layer_id]->set_param_propagate_down(param_id,
                                                      param_need_backward);
        }
        
        for (int param_id = 0; param_id < num_param_blobs; ++param_id) 
        {
         
            // 添加parameter blob,如果当前layer没有parameter blob(num_param_blobs==0),
            // 比如RELU，那么就不进入循环，不添加parameter blob  
            // AppendParam只是执行为当前layer添加parameter blob的相关工作，并不
            // 会修改与backward的相关属性  
            AppendParam(param, layer_id, param_id);
        }

        // 在这里初始化向量layer_need_backward_  
        // Finally, set the backward flag
        layer_need_backward_.push_back(need_backward);

        // 在上述的AppendTop函数中，在遍历当前层的每一个top blob的时候都会将一个false（默认值）压入向量blob_need_backward_。在下面的代码中，如果这个layer need backward，则会更新blob_need_backward_  
        if (need_backward) 
        {
            for (int top_id = 0; top_id < top_id_vecs_[layer_id].size(); ++top_id) 
            {
                // 重新设置每一层的 blob_need_backward_ ，一开始是在AppendTop里将各 top blob 默认设置为false，这里根据need_backward来重新设置  
                blob_need_backward_[top_id_vecs_[layer_id][top_id]] = true;
            }
        }
    }

    // Go through the net backwards to determine which blobs contribute to the  
    // loss.  We can skip backward computation for blobs that don't contribute  
    // to the loss. 不仅仅是确定某个layer是否需要BP，还需要确定layer的某些blob
    // 是否需要BP  
    // Also checks if all bottom blobs don't need backward computation (possible  
    // because the skip_propagate_down param) and so we can skip bacward  
    // computation for the entire layer  
    // 需要注意的是，上述代码中关于backward设置的部分，是按照前向的顺序设置的，
    // 而下面的代码是按后向顺序修正前向设置的结果。  
    // 一个layer是否需要backward computation，主要依据两个方面：(1)该layer的top
    // blob 是否参与loss的计算；(2):该layer是否至少有一个 bottom blob 需要
    // backward computation，比如Data层一般就不需要backward computation  
    set<string> blobs_under_loss;
    set<string> blobs_skip_backp;

    // 为true，则表示当前layer的bottom blob不需要backward computation，即该层不
    // 需要backward computation。  
    // 这个局部变量所表示的意义与caffe.proto里message Layerparameter的
    // propagate_down的定义恰好相反。

    for (int layer_id = layers_.size() - 1; layer_id >= 0; --layer_id) 
    {
        bool layer_contributes_loss = false;
        bool layer_skip_propagate_down = true;
        
        for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) 
        {
        
            // 如果在网络的定义文件文件中，没有设置loss_weight, 那么loss layer
            // 的LayerSetUp方法会设置loss_weight,且默认值为1  
            const string& blob_name = blob_names_[top_id_vecs_[layer_id][top_id]];

            if (layers_[layer_id]->loss(top_id) 
             || (blobs_under_loss.find(blob_name) != blobs_under_loss.end())) 
            {
                layer_contributes_loss = true;
            }
            
            if (blobs_skip_backp.find(blob_name) == blobs_skip_backp.end()) 
            {
                // 只要有一个top blob不在 blobs_skip_backp 里面，
                // layer_skip_propagate_down就为false，即该层不会跳过BP  
                layer_skip_propagate_down = false;
            }
            
            if (layer_contributes_loss && !layer_skip_propagate_down)
            break;
        
        }
        
        // If this layer can skip backward computation, also all his bottom blobs
        // don't need backpropagation
        if (layer_need_backward_[layer_id] && layer_skip_propagate_down) 
        {
            layer_need_backward_[layer_id] = false;

            // <5> 存每一层的输入blob指针
            // 存每一层输入(bottom)的id vector<vector<int> > bottom_id_vecs_ 
            for (int bottom_id = 0; bottom_id < bottom_vecs_[layer_id].size();
               ++bottom_id) 
            {
                bottom_need_backward_[layer_id][bottom_id] = false;
            }
        }
        
        if (!layer_contributes_loss) 
        {
            layer_need_backward_[layer_id] = false; 
        }

        if (Caffe::root_solver()) 
        {
            if (layer_need_backward_[layer_id]) 
            {
                LOG(INFO) << layer_names_[layer_id] << " needs backward computation.";
            } 
            else 
            {
                LOG(INFO) << layer_names_[layer_id]
                << " does not need backward computation.";
            }
        }
        
        // 修正前向设置的结果
        for (int bottom_id = 0; 
             bottom_id < bottom_vecs_[layer_id].size();  
             ++bottom_id) 
        {
            if (layer_contributes_loss) 
            {
                const string& blob_name =
                blob_names_[bottom_id_vecs_[layer_id][bottom_id]];

                // 为blobs_under_loss添加新元素
                blobs_under_loss.insert(blob_name);  
            } 
            else 
            {
                bottom_need_backward_[layer_id][bottom_id] = false;
            }
            
            if (!bottom_need_backward_[layer_id][bottom_id]) 
            {
                const string& blob_name =
                       blob_names_[bottom_id_vecs_[layer_id][bottom_id]];

                // 为blobs_skip_backp添加新元素 
                blobs_skip_backp.insert(blob_name); 
            }
        }
    }

    // 4. param.force_backward()压到blob_need_backward_中 
    // Handle force_backward if needed.Netparameter类型的force_backward方法  
    //  vector<bool> blob_need_backward_
    // Handle force_backward if needed.
    if (param.force_backward()) 
    {
        for (int layer_id = 0; layer_id < layers_.size(); ++layer_id) 
        {
            layer_need_backward_[layer_id] = true;

            for (int bottom_id = 0;
               bottom_id < bottom_need_backward_[layer_id].size(); ++bottom_id) 
            {
                bottom_need_backward_[layer_id][bottom_id] =
                    bottom_need_backward_[layer_id][bottom_id] ||
                    layers_[layer_id]->AllowForceBackward(bottom_id);
                
                blob_need_backward_[bottom_id_vecs_[layer_id][bottom_id]] =
                    blob_need_backward_[bottom_id_vecs_[layer_id][bottom_id]] ||
                    bottom_need_backward_[layer_id][bottom_id];
            }
               
            for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
               ++param_id) 
            {
                layers_[layer_id]->set_param_propagate_down(param_id, true);
            }
        }
    }
    
    // In the end, all remaining blobs are considered output blobs.
    for (set<string>::iterator it = available_blobs.begin();
      it != available_blobs.end(); ++it)
    {
        LOG_IF(INFO, Caffe::root_solver())
            << "This network produces output " << *it;

        // 读取当前blob的id，其中blob_name_to_idx在输入层初始化过了 
        net_output_blobs_.push_back(blobs_[blob_name_to_idx[*it]].get());
        net_output_blob_indices_.push_back(blob_name_to_idx[*it]);
    }
      
    // 输出当前blob的名字    
    for (size_t blob_id = 0; blob_id < blob_names_.size(); ++blob_id) 
    {
        // 第一次使用向量blob_names_index_,逐一添加元素，是一个map  
        blob_names_index_[blob_names_[blob_id]] = blob_id;
    }
    
    for (size_t layer_id = 0; layer_id < layer_names_.size(); ++layer_id) 
    {
        // 第一次使用向量layer_names_index_，逐一添加元素，是一个map  
        layer_names_index_[layer_names_[layer_id]] = layer_id;
    }
    
    ShareWeights();
    debug_info_ = param.debug_info();
    LOG_IF(INFO, Caffe::root_solver()) << "Network initialization done.";
    
}

/*

*/
// FilterNet()给定当前phase/level/stage，移除指定层 
template <typename Dtype>
void Net<Dtype>::FilterNet(const NetParameter& param,
    NetParameter* param_filtered) {
  NetState net_state(param.state());
  param_filtered->CopyFrom(param);
  param_filtered->clear_layer();
  for (int i = 0; i < param.layer_size(); ++i) {
    // param.layers(i)返回的是关于第当前层的参数： 
    const LayerParameter& layer_param = param.layer(i);
    const string& layer_name = layer_param.name();
    CHECK(layer_param.include_size() == 0 || layer_param.exclude_size() == 0)
          << "Specify either include rules or exclude rules; not both.";
    // If no include rules are specified, the layer is included by default and
    // only excluded if it meets one of the exclude rules.
    bool layer_included = (layer_param.include_size() == 0);
    for (int j = 0; layer_included && j < layer_param.exclude_size(); ++j) {
      if (StateMeetsRule(net_state, layer_param.exclude(j), layer_name)) {
        //如果不包含include，只要meet一个include_size(idx)即可  
        layer_included = false;
      }
    }
    for (int j = 0; !layer_included && j < layer_param.include_size(); ++j) {
      if (StateMeetsRule(net_state, layer_param.include(j), layer_name)) {
        //如果包含include，只要符合一个include_size(idx)即可  
        layer_included = true;
      }
    }
    if (layer_included) {
      param_filtered->add_layer()->CopyFrom(layer_param);
    }
  }
}

/*

*/
// net的state是否满足NetStaterule  
template <typename Dtype>
bool Net<Dtype>::StateMeetsRule(const NetState& state,
    const NetStateRule& rule, const string& layer_name) {
  // Check whether the rule is broken due to phase.
  if (rule.has_phase()) {
      if (rule.phase() != state.phase()) {
        LOG_IF(INFO, Caffe::root_solver())
            << "The NetState phase (" << state.phase()
            << ") differed from the phase (" << rule.phase()
            << ") specified by a rule in layer " << layer_name;
        return false;
      }
  }
  // Check whether the rule is broken due to min level.
  if (rule.has_min_level()) {
    if (state.level() < rule.min_level()) {
      LOG_IF(INFO, Caffe::root_solver())
          << "The NetState level (" << state.level()
          << ") is above the min_level (" << rule.min_level()
          << ") specified by a rule in layer " << layer_name;
      return false;
    }
  }
  // Check whether the rule is broken due to max level.
  if (rule.has_max_level()) {
    if (state.level() > rule.max_level()) {
      LOG_IF(INFO, Caffe::root_solver())
          << "The NetState level (" << state.level()
          << ") is above the max_level (" << rule.max_level()
          << ") specified by a rule in layer " << layer_name;
      return false;
    }
  }
  // Check whether the rule is broken due to stage. The NetState must
  // contain ALL of the rule's stages to meet it.
  for (int i = 0; i < rule.stage_size(); ++i) {
    // Check that the NetState contains the rule's ith stage.
    bool has_stage = false;
    for (int j = 0; !has_stage && j < state.stage_size(); ++j) {
      if (rule.stage(i) == state.stage(j)) { has_stage = true; }
    }
    if (!has_stage) {
      LOG_IF(INFO, Caffe::root_solver())
          << "The NetState did not contain stage '" << rule.stage(i)
          << "' specified by a rule in layer " << layer_name;
      return false;
    }
  }
  // Check whether the rule is broken due to not_stage. The NetState must
  // contain NONE of the rule's not_stages to meet it.
  for (int i = 0; i < rule.not_stage_size(); ++i) {
    // Check that the NetState contains the rule's ith not_stage.
    bool has_stage = false;
    for (int j = 0; !has_stage && j < state.stage_size(); ++j) {
      if (rule.not_stage(i) == state.stage(j)) { has_stage = true; }
    }
    if (has_stage) {
      LOG_IF(INFO, Caffe::root_solver())
          << "The NetState contained a not_stage '" << rule.not_stage(i)
          << "' specified by a rule in layer " << layer_name;
      return false;
    }
  }
  return true;
}

/*

*/
// Helper for Net::Init: add a new top blob to the net.
template <typename Dtype>
void Net<Dtype>::AppendTop(const NetParameter& param, const int layer_id,
                           const int top_id, set<string>* available_blobs,
                           map<string, int>* blob_name_to_idx) {
  shared_ptr<LayerParameter> layer_param(
      new LayerParameter(param.layer(layer_id)));
  const string& blob_name = (layer_param->top_size() > top_id) ?
      layer_param->top(top_id) : "(automatic)";
  // Check if we are doing in-place computation
  if (blob_name_to_idx && layer_param->bottom_size() > top_id &&
      blob_name == layer_param->bottom(top_id)) {
    // In-place computation
    LOG_IF(INFO, Caffe::root_solver())
        << layer_param->name() << " -> " << blob_name << " (in-place)";

    // 存每一层输出(top)的blob 
    top_vecs_[layer_id].push_back(blobs_[(*blob_name_to_idx)[blob_name]].get());
    top_id_vecs_[layer_id].push_back((*blob_name_to_idx)[blob_name]);
  } else if (blob_name_to_idx &&
             blob_name_to_idx->find(blob_name) != blob_name_to_idx->end()) {
             // 判断是否进行同址计算
    // If we are not doing in-place computation but have duplicated blobs,
    // raise an error.
    LOG(FATAL) << "Top blob '" << blob_name
               << "' produced by multiple sources.";
  } else {

    // Normal output.
    if (Caffe::root_solver()) {
            // 输出当前blob的名字
      LOG(INFO) << layer_param->name() << " -> " << blob_name;
    }

    // 定义一块新的blob空间，用blob_pointer指向这块空间
    shared_ptr<Blob<Dtype> > blob_pointer(new Blob<Dtype>());

    // blobs只是存储中间结果；每次遍历到一个top blob都会更新blob_id  
    const int blob_id = blobs_.size();

    // 把这个指针存入到blobs_中
    blobs_.push_back(blob_pointer);

    // 把blob_name、force_backward、idx存入对应的容器中
    blob_names_.push_back(blob_name);
    blob_need_backward_.push_back(false);

    //blob_name_to_idx是一个局部变量，其实它是在当前layer的top blob 和下一层的bottom blob间起着一个桥梁作用。  
    //blob_name_to_idx中元素的pair是从网络最开始一层一层搭建的过程中压入map的，其中的name和id都是不重复的。name是关键字——不重复是map数据结构的必然要求，id也是不重复的——0,1,2...  
    //blob_name_to_idx和blobs_一样，在"Normal output"的情形下，每次遍历到一个top blob的时候都会更新 
    if (blob_name_to_idx) { (*blob_name_to_idx)[blob_name] = blob_id; } //添加新元素-->map可以通过下标访问符为（关联）容器添加新元素 blob_name_to_idx是指针，并不是空指针，所以可以执行if之后的代码。  
    top_id_vecs_[layer_id].push_back(blob_id);
    top_vecs_[layer_id].push_back(blob_pointer.get());
  }

  // 向available_blobs插入当前blob的名字
  if (available_blobs) { available_blobs->insert(blob_name); }
}

/*

*/
// Helper for Net::Init: add a new bottom blob to the net.
template <typename Dtype>
int Net<Dtype>::AppendBottom(const NetParameter& param, const int layer_id,
    const int bottom_id, set<string>* available_blobs,
    map<string, int>* blob_name_to_idx) {
  const LayerParameter& layer_param = param.layer(layer_id);

  // layer_param.bottom(j)存的是第j个输入blob的名字
  const string& blob_name = layer_param.bottom(bottom_id);
  if (available_blobs->find(blob_name) == available_blobs->end()) {
    LOG(FATAL) << "Unknown bottom blob '" << blob_name << "' (layer '"
               << layer_param.name() << "', bottom index " << bottom_id << ")";
  }

  // blob_name_to_idx是一个map,其关键字是不重复的。blob_name_to_idx在输入层初始化过了-->*blob_name_to_idx)[blob_name] = blob_id  
  const int blob_id = (*blob_name_to_idx)[blob_name];
  LOG_IF(INFO, Caffe::root_solver())
      << layer_names_[layer_id] << " <- " << blob_name;

  //调用shared_ptr类的get()方法提取存储在blobs_中的中间变量  
  // 存入第j个输入blob的指针
  bottom_vecs_[layer_id].push_back(blobs_[blob_id].get());

  // 存入第j个输入blob的id
  bottom_id_vecs_[layer_id].push_back(blob_id);

  // 从available_blobs中删除第j个blob的名字
  available_blobs->erase(blob_name);
  bool propagate_down = true; // propagate_down默认为true  
  // Check if the backpropagation on bottom_id should be skipped
  if (layer_param.propagate_down_size() > 0)
    propagate_down = layer_param.propagate_down(bottom_id);

  //propagate_down为true,则表示参与BP;否则，skip bp    
  // 更新need_backward
  const bool need_backward = blob_need_backward_[blob_id] &&
                          propagate_down;
  bottom_need_backward_[layer_id].push_back(need_backward);
  return blob_id;
}

/*

*/
template <typename Dtype>
void Net<Dtype>::AppendParam(const NetParameter& param, const int layer_id,
                             const int param_id) {
  //模板类Layer的layer_param方法，返回Layerparameter类型成员  
  const LayerParameter& layer_param = layers_[layer_id]->layer_param();
  const int param_size = layer_param.param_size();
  string param_name =
      (param_size > param_id) ? layer_param.param(param_id).name() : "";
  if (param_name.size()) {
    //vector<string> param_display_names_ 这里param_name获取的是PaParamSpec类型中的name成员，如果有name且非空,就把name压入该向量，否则就压入param_id  
    param_display_names_.push_back(param_name);//vector<shared_ptr<Blob<Dtype> > > params_--->The parameters in the network,整个网络的参数的id,!!!不管这个参数有没有non-emty name，是否参与share!!!  
  } else {
    ostringstream param_display_name;
    param_display_name << param_id;
    param_display_names_.push_back(param_display_name.str());
  }
//Append 参数blob 每一次循环，net_param_id和param_id_vecs_都会更新  
  const int net_param_id = params_.size();
//将当前layer当前"参数blob"压入params_ --->vector<shared_ptr<Blob<Dtype> > > params_  
  params_.push_back(layers_[layer_id]->blobs()[param_id]);
//将整个网络的参数按层的形式来存储，存储的元素可以理解为params_这个向量的下标值（类型为整型）  
  param_id_vecs_[layer_id].push_back(net_param_id);
//param_layer_indices_是向量，其元素为当layer_id 与当前param_id 组成的pair.vector<pair<int, int> > param_layer_indices_  
  param_layer_indices_.push_back(make_pair(layer_id, param_id));

 //获取每个param_id所对应的Paramspec类型成员，如果param_id >= param_size 则返回default_param_spec。注意param_size <= num_param_blobs  
  ParamSpec default_param_spec;
  const ParamSpec* param_spec = (layer_param.param_size() > param_id) ?
      &layer_param.param(param_id) : &default_param_spec;
  if (!param_size || !param_name.size() || (param_name.size() &&
      param_names_index_.find(param_name) == param_names_index_.end())) {

    // 相反，如果param_name不为空，而且能够在param_names_index_中找到，说明这个parameter已经存在于之前的某个或者某些网络层里，说明这个parameter是共享于多个layer  
    // 在caffe.proto的message ParamSpec里关于name的注释——>To share a parameter between two layers, give it a (non-empty) name, 可见，如果一个parameter是共享与多个网络层，那么它会有一个非空的name  
    // This layer "owns" this parameter blob -- it is either anonymous
    // (i.e., not given a param_name) or explicitly given a name that we
    // haven't already seen.
    param_owners_.push_back(-1);//vector<int> param_owners_ 是一个存储parameter "onwer"的一个向量  ——> -1 表示当前Layer就是该parameter的"owner"  

  //添加param_name  
    if (param_name.size()) {
           //map<string, int> param_names_index_是整个网络的参数non-empty name与index的映射。  
      //注意，这个name是ParamSpec 类型中的name,而且，""To share a parameter between two layers, give it a (non-empty) name"",所以说这个map中存储的pair是<会被share的parameter_name, 其对应index>     
    //map<string, int> param_names_index_ 。虽然每一次循环，net_param_id都会更新，但是net_param_id只有当param_name.size()>0时才会被压入向量param_names_index_ 
       param_names_index_[param_name] = net_param_id;
    }

    //添加learnable_param 
    //vector<Blob<Dtype>*> learnable_params_   
    const int learnable_param_id = learnable_params_.size();

    //压入learnable parameter ---> 在模板类layer中，定义了一个blobs_成员，其存储的就是learnable parameter。随后压入learnable_param_id  
    learnable_params_.push_back(params_[net_param_id].get());
    learnable_param_ids_.push_back(learnable_param_id); //vector<int> learnable_param_ids_ 
    has_params_lr_.push_back(param_spec->has_lr_mult()); //vector<bool> has_params_lr_  
    has_params_decay_.push_back(param_spec->has_decay_mult());
    params_lr_.push_back(param_spec->lr_mult()); //vector<float> params_lr_  
    params_weight_decay_.push_back(param_spec->decay_mult());
  } else {
//因为"To share a parameter between two layers, give it a (non-empty) name",所以这句代码就是获取shared parameter的"owner" net_param_id  
    // Named param blob with name we've seen before: share params
    const int owner_net_param_id = param_names_index_[param_name];
    param_owners_.push_back(owner_net_param_id);//vector<int> param_owners_ 

    //只获取了那些shared的parameter,即具有non-empty name的parameter的pair<layer_id, param_id>  
    const pair<int, int>& owner_index =
        param_layer_indices_[owner_net_param_id];
    const int owner_layer_id = owner_index.first;
    const int owner_param_id = owner_index.second;
    LOG_IF(INFO, Caffe::root_solver()) << "Sharing parameters '" << param_name
        << "' owned by "
        << "layer '" << layer_names_[owner_layer_id] << "', param "
        << "index " << owner_param_id;

    //获取当前层的当前参数Blob  
    Blob<Dtype>* this_blob = layers_[layer_id]->blobs()[param_id].get();

    // 获取owner layer的对应的参数blob  
    Blob<Dtype>* owner_blob =
        layers_[owner_layer_id]->blobs()[owner_param_id].get();
    const int param_size = layer_param.param_size();
    if (param_size > param_id && (layer_param.param(param_id).share_mode() ==
                                  ParamSpec_DimCheckMode_PERMISSIVE)) {
      // Permissive dimension checking -- only check counts are the same.
      CHECK_EQ(this_blob->count(), owner_blob->count())
          << "Cannot share param '" << param_name << "' owned by layer '"
          << layer_names_[owner_layer_id] << "' with layer '"
          << layer_names_[layer_id] << "'; count mismatch.  Owner layer param "
          << "shape is " << owner_blob->shape_string() << "; sharing layer "
          << "shape is " << this_blob->shape_string();
    } else {
      // Strict dimension checking -- all dims must be the same.
      CHECK(this_blob->shape() == owner_blob->shape())
          << "Cannot share param '" << param_name << "' owned by layer '"
          << layer_names_[owner_layer_id] << "' with layer '"
          << layer_names_[layer_id] << "'; shape mismatch.  Owner layer param "
          << "shape is " << owner_blob->shape_string() << "; sharing layer "
          << "expects shape " << this_blob->shape_string();
    }

    //获取owner layer的learnable_param_id，并且压入当前layer的向量learnable_param_ids_。  
    //而且在这里也没有把参数blob压入learnable_params_向量（只是将id压入learnable_param_ids_），从而避免当前layer与sharing layer之间关于shared parameter blob 的重复  
    const int learnable_param_id = learnable_param_ids_[owner_net_param_id];
    learnable_param_ids_.push_back(learnable_param_id);
    if (param_spec->has_lr_mult()) {
      if (has_params_lr_[learnable_param_id]) {
        CHECK_EQ(param_spec->lr_mult(), params_lr_[learnable_param_id])
            << "Shared param '" << param_name << "' has mismatched lr_mult.";
      } else {
        has_params_lr_[learnable_param_id] = true;
        params_lr_[learnable_param_id] = param_spec->lr_mult();
      }
    }
    if (param_spec->has_decay_mult()) {
      if (has_params_decay_[learnable_param_id]) {
        CHECK_EQ(param_spec->decay_mult(),
                 params_weight_decay_[learnable_param_id])
            << "Shared param '" << param_name << "' has mismatched decay_mult.";
      } else {
        has_params_decay_[learnable_param_id] = true;
        params_weight_decay_[learnable_param_id] = param_spec->decay_mult();
      }
    }
  }
}

/*

*/
template <typename Dtype>
Dtype Net<Dtype>::ForwardFromTo(int start, int end) 
{
    CHECK_GE(start, 0);
    CHECK_LT(end, layers_.size());
    Dtype loss = 0;
    
    for (int i = start; i <= end; ++i) 
    {
        // top_vecs_[i]对于第i层，插入当前blob的指针
        // LOG(ERROR) << "Forwarding " << layer_names_[i];
        Dtype layer_loss = layers_[i]->Forward(bottom_vecs_[i], top_vecs_[i]);
        loss += layer_loss;
        
        if (debug_info_) 
        { 
            ForwardDebugInfo(i); 
        }
    }

    // 对于非loss层都会返回0
    return loss;
}

template <typename Dtype>
Dtype Net<Dtype>::ForwardFrom(int start) {
  return ForwardFromTo(start, layers_.size() - 1);
}

template <typename Dtype>
Dtype Net<Dtype>::ForwardTo(int end) {
  return ForwardFromTo(0, end);
}

/*

*/
template <typename Dtype>
const vector<Blob<Dtype>*>& Net<Dtype>::Forward(Dtype* loss) 
{
    if (loss != NULL) 
    {
        *loss = ForwardFromTo(0, layers_.size() - 1);
    } 
    else 
    {
        ForwardFromTo(0, layers_.size() - 1);
    }
    
    return net_output_blobs_;
}

/*

*/
// 把网络输入层的blob读到net_input_blobs_，然后进行前馈，计算出loss。Forward的
// 重载，只是输入层的blob以string的格式传入。 
template <typename Dtype>
const vector<Blob<Dtype>*>& Net<Dtype>::Forward(
    const vector<Blob<Dtype>*> & bottom, Dtype* loss) 
{
    LOG_EVERY_N(WARNING, 1000) << "DEPRECATED: Forward(bottom, loss) "
      << "will be removed in a future version. Use Forward(loss).";

    // 从上面solver.cpp可见bottom.size()为0
    // Copy bottom to net bottoms
    for (int i = 0; i < bottom.size(); ++i) 
    {
        net_input_blobs_[i]->CopyFrom(*bottom[i]);
    }
    
    return Forward(loss);
}

/*

*/
template <typename Dtype>
void Net<Dtype>::BackwardFromTo(int start, int end) {
  CHECK_GE(end, 0);
  CHECK_LT(start, layers_.size());

  // 一般情况下，第一个卷积层conv1的propagatedown为false，即bottom_need_backward_[0]为false， 也就是说不需要求关于conv1的bottom blob的梯度，因为这些bottom blob是data，label， 它们毕竟是死的，不变的，不会随着模型的学习而改变  
  for (int i = start; i >= end; --i) {
    if (layer_need_backward_[i]) {
      layers_[i]->Backward(
          top_vecs_[i], bottom_need_backward_[i], bottom_vecs_[i]);
      if (debug_info_) { BackwardDebugInfo(i); }
    }
  }
}

/*

*/
template <typename Dtype>
void Net<Dtype>::ForwardDebugInfo(const int layer_id) {
  for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
    const Blob<Dtype>& blob = *top_vecs_[layer_id][top_id];
    const string& blob_name = blob_names_[top_id_vecs_[layer_id][top_id]];
    const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Forward] "
        << "Layer " << layer_names_[layer_id]
        << ", top blob " << blob_name
        << " data: " << data_abs_val_mean;
  }
  for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
       ++param_id) {
    const Blob<Dtype>& blob = *layers_[layer_id]->blobs()[param_id];
    const int net_param_id = param_id_vecs_[layer_id][param_id];
    const string& blob_name = param_display_names_[net_param_id];
    const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Forward] "
        << "Layer " << layer_names_[layer_id]
        << ", param blob " << blob_name
        << " data: " << data_abs_val_mean;
  }
}

template <typename Dtype>
void Net<Dtype>::BackwardDebugInfo(const int layer_id) {
  const vector<Blob<Dtype>*>& bottom_vec = bottom_vecs_[layer_id];
  for (int bottom_id = 0; bottom_id < bottom_vec.size(); ++bottom_id) {
    if (!bottom_need_backward_[layer_id][bottom_id]) { continue; }
    const Blob<Dtype>& blob = *bottom_vec[bottom_id];
    const string& blob_name = blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
    const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Backward] "
        << "Layer " << layer_names_[layer_id]
        << ", bottom blob " << blob_name
        << " diff: " << diff_abs_val_mean;
  }
  for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
       ++param_id) {
    if (!layers_[layer_id]->param_propagate_down(param_id)) { continue; }
    const Blob<Dtype>& blob = *layers_[layer_id]->blobs()[param_id];
    const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Backward] "
        << "Layer " << layer_names_[layer_id]
        << ", param blob " << param_id
        << " diff: " << diff_abs_val_mean;
  }
}

template <typename Dtype>
void Net<Dtype>::UpdateDebugInfo(const int param_id) {
  const Blob<Dtype>& blob = *params_[param_id];
  const int param_owner = param_owners_[param_id];
  const string& layer_name = layer_names_[param_layer_indices_[param_id].first];
  const string& param_display_name = param_display_names_[param_id];
  const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
  if (param_owner < 0) {
    const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Update] Layer " << layer_name
        << ", param " << param_display_name
        << " data: " << data_abs_val_mean
        << "; diff: " << diff_abs_val_mean;
  } else {
    const string& owner_layer_name =
        layer_names_[param_layer_indices_[param_owner].first];
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Update] Layer " << layer_name
        << ", param blob " << param_display_name
        << " (owned by layer " << owner_layer_name << ", " << "param "
        << param_display_names_[param_owners_[param_id]] << ")"
        << " diff: " << diff_abs_val_mean;
  }
}
/*
功能：从Other网络复制某些层 
步骤：对Other网络的第i层（源层）： 
1. 定义一个Layer的指针指向第i层 
2. 读取第i层（源层）的名字 
3. 找通过名字来找目标层 
如果没找到，即target_layer_id == layer_names_.size() 
则忽略Other的第i层，即Other的第i层不需要share给网络 
4. 如果找到了，即other的第i层需要share给网络， 
则把目标层的所有blob读到target_blobs中

判断目标层和源层的blob数量是否相等
判断每个blob大小是否相等
调用ShareData函数把源层的blob赋给目标层的blob
*/

template <typename Dtype>
void Net<Dtype>::ShareTrainedLayersWith(const Net* other) {
  int num_source_layers = other->layers().size();
  for (int i = 0; i < num_source_layers; ++i) {
    Layer<Dtype>* source_layer = other->layers()[i].get();
    const string& source_layer_name = other->layer_names()[i];
    int target_layer_id = 0;
    while (target_layer_id != layer_names_.size() &&
        layer_names_[target_layer_id] != source_layer_name) {
      ++target_layer_id;
    }
    if (target_layer_id == layer_names_.size()) {
      LOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    DLOG(INFO) << "Copying source layer " << source_layer_name;
    vector<shared_ptr<Blob<Dtype> > >& target_blobs =
        layers_[target_layer_id]->blobs();
    CHECK_EQ(target_blobs.size(), source_layer->blobs().size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    for (int j = 0; j < target_blobs.size(); ++j) {
      Blob<Dtype>* source_blob = source_layer->blobs()[j].get();
      CHECK(target_blobs[j]->shape() == source_blob->shape())
          << "Cannot share param " << j << " weights from layer '"
          << source_layer_name << "'; shape mismatch.  Source param shape is "
          << source_blob->shape_string() << "; target param shape is "
          << target_blobs[j]->shape_string();
      target_blobs[j]->ShareData(*source_blob);
    }
  }
}

template <typename Dtype>
void Net<Dtype>::BackwardFrom(int start) {
  BackwardFromTo(start, 0);
}

template <typename Dtype>
void Net<Dtype>::BackwardTo(int end) {
  BackwardFromTo(layers_.size() - 1, end);
}
// 对整个网络进行反向传播。
template <typename Dtype>
void Net<Dtype>::Backward() {
  BackwardFromTo(layers_.size() - 1, 0);
  if (debug_info_) {
    Dtype asum_data = 0, asum_diff = 0, sumsq_data = 0, sumsq_diff = 0;
    for (int i = 0; i < learnable_params_.size(); ++i) {
      asum_data += learnable_params_[i]->asum_data();
      asum_diff += learnable_params_[i]->asum_diff();
      sumsq_data += learnable_params_[i]->sumsq_data();
      sumsq_diff += learnable_params_[i]->sumsq_diff();
    }
    const Dtype l2norm_data = std::sqrt(sumsq_data);
    const Dtype l2norm_diff = std::sqrt(sumsq_diff);
    LOG(ERROR) << "    [Backward] All net params (data, diff): "
               << "L1 norm = (" << asum_data << ", " << asum_diff << "); "
               << "L2 norm = (" << l2norm_data << ", " << l2norm_diff << ")";
  }
}
// 用于改变每层的尺寸，比如输出的feature map的size  
template <typename Dtype>
void Net<Dtype>::Reshape() {
  for (int i = 0; i < layers_.size(); ++i) {
    layers_[i]->Reshape(bottom_vecs_[i], top_vecs_[i]);
  }
}
/*
功能：和ShareTrainedLayersWith一样 
步骤：不同的是调用FromProto函数把源层的blob赋给目标层的blob
*/
template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFrom(const NetParameter& param) {
  int num_source_layers = param.layer_size();
  for (int i = 0; i < num_source_layers; ++i) {
    const LayerParameter& source_layer = param.layer(i);
    const string& source_layer_name = source_layer.name();
    int target_layer_id = 0;
    while (target_layer_id != layer_names_.size() &&
        layer_names_[target_layer_id] != source_layer_name) {
      ++target_layer_id;
    }
    if (target_layer_id == layer_names_.size()) {
      LOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    DLOG(INFO) << "Copying source layer " << source_layer_name;
    vector<shared_ptr<Blob<Dtype> > >& target_blobs =
        layers_[target_layer_id]->blobs();
    CHECK_EQ(target_blobs.size(), source_layer.blobs_size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    for (int j = 0; j < target_blobs.size(); ++j) {
      if (!target_blobs[j]->ShapeEquals(source_layer.blobs(j))) {
        Blob<Dtype> source_blob;
        const bool kReshape = true;
        source_blob.FromProto(source_layer.blobs(j), kReshape);
        LOG(FATAL) << "Cannot copy param " << j << " weights from layer '"
            << source_layer_name << "'; shape mismatch.  Source param shape is "
            << source_blob.shape_string() << "; target param shape is "
            << target_blobs[j]->shape_string() << ". "
            << "To learn this layer's parameters from scratch rather than "
            << "copying from a saved net, rename the layer.";
      }
      const bool kReshape = false;
      target_blobs[j]->FromProto(source_layer.blobs(j), kReshape);
    }
  }
}
// 功能：从文件中读入NetParameter param，然后调用CopyTrainedLayersFrom()
// 调用FromProto函数把源层的blob赋给目标层的blob。 
template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFrom(const string trained_filename) {
  if (trained_filename.size() >= 3 &&
      trained_filename.compare(trained_filename.size() - 3, 3, ".h5") == 0) {
    CopyTrainedLayersFromHDF5(trained_filename);
  } else {
    CopyTrainedLayersFromBinaryProto(trained_filename);
  }
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFromBinaryProto(
    const string trained_filename) {
  NetParameter param;
  ReadNetParamsFromBinaryFileOrDie(trained_filename, &param);
  CopyTrainedLayersFrom(param);
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFromHDF5(const string trained_filename) {
  hid_t file_hid = H5Fopen(trained_filename.c_str(), H5F_ACC_RDONLY,
                           H5P_DEFAULT);
  CHECK_GE(file_hid, 0) << "Couldn't open " << trained_filename;
  hid_t data_hid = H5Gopen2(file_hid, "data", H5P_DEFAULT);
  CHECK_GE(data_hid, 0) << "Error reading weights from " << trained_filename;
  int num_layers = hdf5_get_num_links(data_hid);
  for (int i = 0; i < num_layers; ++i) {
    string source_layer_name = hdf5_get_name_by_idx(data_hid, i);
    if (!layer_names_index_.count(source_layer_name)) {
      LOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    int target_layer_id = layer_names_index_[source_layer_name];
    DLOG(INFO) << "Copying source layer " << source_layer_name;
    vector<shared_ptr<Blob<Dtype> > >& target_blobs =
        layers_[target_layer_id]->blobs();
    hid_t layer_hid = H5Gopen2(data_hid, source_layer_name.c_str(),
        H5P_DEFAULT);
    CHECK_GE(layer_hid, 0)
        << "Error reading weights from " << trained_filename;
    // Check that source layer doesn't have more params than target layer
    int num_source_params = hdf5_get_num_links(layer_hid);
    CHECK_LE(num_source_params, target_blobs.size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    for (int j = 0; j < target_blobs.size(); ++j) {
      ostringstream oss;
      oss << j;
      string dataset_name = oss.str();
      int target_net_param_id = param_id_vecs_[target_layer_id][j];
      if (!H5Lexists(layer_hid, dataset_name.c_str(), H5P_DEFAULT)) {
        // Target param doesn't exist in source weights...
        if (param_owners_[target_net_param_id] != -1) {
          // ...but it's weight-shared in target, so that's fine.
          continue;
        } else {
          LOG(FATAL) << "Incompatible number of blobs for layer "
              << source_layer_name;
        }
      }
      hdf5_load_nd_dataset(layer_hid, dataset_name.c_str(), 0, kMaxBlobAxes,
          target_blobs[j].get());
    }
    H5Gclose(layer_hid);
  }
  H5Gclose(data_hid);
  H5Fclose(file_hid);
}
// 把网络的参数存入prototxt中。  
template <typename Dtype>
void Net<Dtype>::ToProto(NetParameter* param, bool write_diff) const {
  param->Clear();
  param->set_name(name_);
  // Add bottom and top
  DLOG(INFO) << "Serializing " << layers_.size() << " layers";
  for (int i = 0; i < layers_.size(); ++i) {
    LayerParameter* layer_param = param->add_layer();
    layers_[i]->ToProto(layer_param, write_diff);
  }
}

template <typename Dtype>
void Net<Dtype>::ToHDF5(const string& filename, bool write_diff) const {
  hid_t file_hid = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
      H5P_DEFAULT);
  CHECK_GE(file_hid, 0)
      << "Couldn't open " << filename << " to save weights.";
  hid_t data_hid = H5Gcreate2(file_hid, "data", H5P_DEFAULT, H5P_DEFAULT,
      H5P_DEFAULT);
  CHECK_GE(data_hid, 0) << "Error saving weights to " << filename << ".";
  hid_t diff_hid = -1;
  if (write_diff) {
    diff_hid = H5Gcreate2(file_hid, "diff", H5P_DEFAULT, H5P_DEFAULT,
        H5P_DEFAULT);
    CHECK_GE(diff_hid, 0) << "Error saving weights to " << filename << ".";
  }
  for (int layer_id = 0; layer_id < layers_.size(); ++layer_id) {
    const LayerParameter& layer_param = layers_[layer_id]->layer_param();
    string layer_name = layer_param.name();
    hid_t layer_data_hid = H5Gcreate2(data_hid, layer_name.c_str(),
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    CHECK_GE(layer_data_hid, 0)
        << "Error saving weights to " << filename << ".";
    hid_t layer_diff_hid = -1;
    if (write_diff) {
      layer_diff_hid = H5Gcreate2(diff_hid, layer_name.c_str(),
          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      CHECK_GE(layer_diff_hid, 0)
          << "Error saving weights to " << filename << ".";
    }
    int num_params = layers_[layer_id]->blobs().size();
    for (int param_id = 0; param_id < num_params; ++param_id) {
      ostringstream dataset_name;
      dataset_name << param_id;
      const int net_param_id = param_id_vecs_[layer_id][param_id];
      if (param_owners_[net_param_id] == -1) {
        // Only save params that own themselves
        hdf5_save_nd_dataset<Dtype>(layer_data_hid, dataset_name.str(),
            *params_[net_param_id]);
      }
      if (write_diff) {
        // Write diffs regardless of weight-sharing
        hdf5_save_nd_dataset<Dtype>(layer_diff_hid, dataset_name.str(),
            *params_[net_param_id], true);
      }
    }
    H5Gclose(layer_data_hid);
    if (write_diff) {
      H5Gclose(layer_diff_hid);
    }
  }
  H5Gclose(data_hid);
  if (write_diff) {
    H5Gclose(diff_hid);
  }
  H5Fclose(file_hid);
}

// Step() 函数中每次迭代都会调用ApplyUpdate()(class SGDSolver)->Update()(class net)
// 更新params_中blob的值。  
template <typename Dtype>
void Net<Dtype>::Update() {
  for (int i = 0; i < learnable_params_.size(); ++i) {
    learnable_params_[i]->Update();
  }
}

template <typename Dtype>
void Net<Dtype>::ClearParamDiffs() {
  for (int i = 0; i < learnable_params_.size(); ++i) {
    Blob<Dtype>* blob = learnable_params_[i];
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_set(blob->count(), static_cast<Dtype>(0),
                blob->mutable_cpu_diff());
      break;
    case Caffe::GPU:
#ifndef CPU_ONLY
      caffe_gpu_set(blob->count(), static_cast<Dtype>(0),
                    blob->mutable_gpu_diff());
#else
      NO_GPU;
#endif
      break;
    }
  }
}

template <typename Dtype>
void Net<Dtype>::ShareWeights() 
{
    for (int i = 0; i < params_.size(); ++i) 
    {
        if (param_owners_[i] < 0) 
        {
            continue; 
        }
        
        params_[i]->ShareData(*params_[param_owners_[i]]);
        params_[i]->ShareDiff(*params_[param_owners_[i]]);
    }
}

template <typename Dtype>
bool Net<Dtype>::has_blob(const string& blob_name) const {
  return blob_names_index_.find(blob_name) != blob_names_index_.end();
}

template <typename Dtype>
const shared_ptr<Blob<Dtype> > Net<Dtype>::blob_by_name(
    const string& blob_name) const {
  shared_ptr<Blob<Dtype> > blob_ptr;
  if (has_blob(blob_name)) {
    blob_ptr = blobs_[blob_names_index_.find(blob_name)->second];
  } else {
    blob_ptr.reset((Blob<Dtype>*)(NULL));
    LOG(WARNING) << "Unknown blob name " << blob_name;
  }
  return blob_ptr;
}

template <typename Dtype>
bool Net<Dtype>::has_layer(const string& layer_name) const {
  return layer_names_index_.find(layer_name) != layer_names_index_.end();
}

template <typename Dtype>
const shared_ptr<Layer<Dtype> > Net<Dtype>::layer_by_name(
    const string& layer_name) const {
  shared_ptr<Layer<Dtype> > layer_ptr;
  if (has_layer(layer_name)) {
    layer_ptr = layers_[layer_names_index_.find(layer_name)->second];
  } else {
    layer_ptr.reset((Layer<Dtype>*)(NULL));
    LOG(WARNING) << "Unknown layer name " << layer_name;
  }
  return layer_ptr;
}

INSTANTIATE_CLASS(Net);

}  // namespace caffe
