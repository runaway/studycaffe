#include <cstdio>

#include <string>
#include <vector>

#include "caffe/solver.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

namespace caffe {

template<typename Dtype>
void Solver<Dtype>::SetActionFunction(ActionCallback func) {
  action_request_function_ = func;
}

template<typename Dtype>
SolverAction::Enum Solver<Dtype>::GetRequestedAction() {
  if (action_request_function_) {
    // If the external request function has been set, call it.
    return action_request_function_();
  }
  return SolverAction::NONE;
}

// 初始化两个Net类，net_和test_net_，并调用Init()函数 
// 输入：SolverParameter类型的param
template <typename Dtype>
Solver<Dtype>::Solver(const SolverParameter& param, const Solver* root_solver)
    : net_(), callbacks_(), root_solver_(root_solver),
      requested_early_exit_(false) {
  Init(param);
}

// 初始化两个Net类，net_和test_net_，并调用Init()函数 
// 输入：string类型的param_file 
template <typename Dtype>
Solver<Dtype>::Solver(const string& param_file, const Solver* root_solver)
    : net_(), callbacks_(), root_solver_(root_solver),
      requested_early_exit_(false) {
  SolverParameter param;
  ReadSolverParamsFromTextFileOrDie(param_file, &param);
  Init(param);
}

// 初始化网络 
// 输入：SolverParameter类型的param 
// 输出：无
template <typename Dtype>
void Solver<Dtype>::Init(const SolverParameter& param) 
{
    CHECK(Caffe::root_solver() || root_solver_)
      << "root_solver_ needs to be set for all non-root solvers";
    LOG_IF(INFO, Caffe::root_solver()) << "Initializing solver from parameters: "
    << std::endl << param.DebugString();

    // 为solver类的数据成员param_赋值  
    param_ = param;
    CHECK_GE(param_.average_loss(), 1) << "average_loss should be non-negative.";
    CheckSnapshotWritePermissions();

    // 1. 设置随机数种子 
    if (Caffe::root_solver() && param_.random_seed() >= 0) 
    {
        // 调用Caffe命名空间里的set_random_seed函数，而不是caffe类的set_random_seed函数；
        // param_.random_seed()实际上调用的是::google::protobuf::int64 random_seed() 
        Caffe::set_random_seed(param_.random_seed());
    }

    // 2. 申请一块Net空间以下面的构造函数进行初始化 
    // param_file=train_net_，net_指向这块空间 
    // Scaffolding code
    InitTrainNet();

    // 如果有test_net，则申请一块Net空间，test_net_指向这块空间 
    if (Caffe::root_solver()) 
    {
        InitTestNets();
        LOG(INFO) << "Solver scaffolding done.";
    }
    
    iter_ = 0;
    current_step_ = 0;
}

template <typename Dtype>
void Solver<Dtype>::InitTrainNet() {
  const int num_train_nets = param_.has_net() + param_.has_net_param() +
      param_.has_train_net() + param_.has_train_net_param();
  const string& field_names = "net, net_param, train_net, train_net_param";
  CHECK_GE(num_train_nets, 1) << "SolverParameter must specify a train net "
      << "using one of these fields: " << field_names;
  CHECK_LE(num_train_nets, 1) << "SolverParameter must not contain more than "
      << "one of these fields specifying a train_net: " << field_names;
  NetParameter net_param;
  if (param_.has_train_net_param()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net specified in train_net_param.";
    net_param.CopyFrom(param_.train_net_param());
  } else if (param_.has_train_net()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net from train_net file: " << param_.train_net();
    ReadNetParamsFromTextFileOrDie(param_.train_net(), &net_param);
  }
  if (param_.has_net_param()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net specified in net_param.";
    net_param.CopyFrom(param_.net_param());
  }
  if (param_.has_net()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net from net file: " << param_.net();
    ReadNetParamsFromTextFileOrDie(param_.net(), &net_param);
  }
  // Set the correct NetState.  We start with the solver defaults (lowest
  // precedence); then, merge in any NetState specified by the net_param itself;
  // finally, merge in any NetState specified by the train_state (highest
  // precedence).
  NetState net_state;
  net_state.set_phase(TRAIN);

  // 从低到高获取state,最终从最高优先级SolverParameter类型中的train_state,显然这会覆盖掉之前获取的state。  
  net_state.MergeFrom(net_param.state());

  // 这里获取的state可以为Netparameter中的state赋值，然后可以根据LayerParameter中的include和exclude来确定该层是否应该包含在网络中。  
  net_state.MergeFrom(param_.train_state());

  // 这是Initialize train net 的一部分工作。InitTestNets也是如此 
  net_param.mutable_state()->CopyFrom(net_state);

  if (Caffe::root_solver()) {
    // 调用模板类的构造函数，进行net的初始化  
    net_.reset(new Net<Dtype>(net_param));
  } else {
    net_.reset(new Net<Dtype>(net_param, root_solver_->net_.get()));
  }
}

// 需要注意的是TestNet可以有多个，而TrainNet只能有一个
template <typename Dtype>
void Solver<Dtype>::InitTestNets() {
  CHECK(Caffe::root_solver());
  const bool has_net_param = param_.has_net_param();
  const bool has_net_file = param_.has_net();
  const int num_generic_nets = has_net_param + has_net_file;
  CHECK_LE(num_generic_nets, 1)
      << "Both net_param and net_file may not be specified.";
  const int num_test_net_params = param_.test_net_param_size();
  const int num_test_net_files = param_.test_net_size();
  const int num_test_nets = num_test_net_params + num_test_net_files;
  if (num_generic_nets) {
      CHECK_GE(param_.test_iter_size(), num_test_nets)
          << "test_iter must be specified for each test network.";
  } else {
      CHECK_EQ(param_.test_iter_size(), num_test_nets)
          << "test_iter must be specified for each test network.";
  }
  // If we have a generic net (specified by net or net_param, rather than
  // test_net or test_net_param), we may have an unlimited number of actual
  // test networks -- the actual number is given by the number of remaining
  // test_iters after any test nets specified by test_net_param and/or test_net
  // are evaluated.
  // 可以有多个test net  
  const int num_generic_net_instances = param_.test_iter_size() - num_test_nets;

  // num_test_net_instances由num_test_nets和num_generic_net_instances组成，实际上也就是param_.test_iter_size()  
  const int num_test_net_instances = num_test_nets + num_generic_net_instances;
  if (param_.test_state_size()) {
    CHECK_EQ(param_.test_state_size(), num_test_net_instances)
        << "test_state must be unspecified or specified once per test net.";
  }
  if (num_test_net_instances) {
    CHECK_GT(param_.test_interval(), 0);
  }
  int test_net_id = 0;
  vector<string> sources(num_test_net_instances);
  vector<NetParameter> net_params(num_test_net_instances);
  for (int i = 0; i < num_test_net_params; ++i, ++test_net_id) {
      sources[test_net_id] = "test_net_param";
      net_params[test_net_id].CopyFrom(param_.test_net_param(i));
  }
  for (int i = 0; i < num_test_net_files; ++i, ++test_net_id) {
      sources[test_net_id] = "test_net file: " + param_.test_net(i);
      ReadNetParamsFromTextFileOrDie(param_.test_net(i),
          &net_params[test_net_id]);
  }
  const int remaining_test_nets = param_.test_iter_size() - test_net_id;
  if (has_net_param) {
    for (int i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
      sources[test_net_id] = "net_param";
      net_params[test_net_id].CopyFrom(param_.net_param());
    }
  }
  if (has_net_file) {
    for (int i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
      sources[test_net_id] = "net file: " + param_.net();
      ReadNetParamsFromTextFileOrDie(param_.net(), &net_params[test_net_id]);
    }
  }
  test_nets_.resize(num_test_net_instances);
  for (int i = 0; i < num_test_net_instances; ++i) {
    // Set the correct NetState.  We start with the solver defaults (lowest
    // precedence); then, merge in any NetState specified by the net_param
    // itself; finally, merge in any NetState specified by the test_state
    // (highest precedence).
    NetState net_state;
    net_state.set_phase(TEST);
    net_state.MergeFrom(net_params[i].state());
    if (param_.test_state_size()) {
      net_state.MergeFrom(param_.test_state(i));
    }
    net_params[i].mutable_state()->CopyFrom(net_state);
    LOG(INFO)
        << "Creating test net (#" << i << ") specified by " << sources[i];
    if (Caffe::root_solver()) {
      test_nets_[i].reset(new Net<Dtype>(net_params[i]));
    } else {
      test_nets_[i].reset(new Net<Dtype>(net_params[i],
          root_solver_->test_nets_[i].get()));
    }
    test_nets_[i]->set_debug_info(param_.debug_info());
  }
}

template <typename Dtype>
void Solver<Dtype>::Step(int iters) 
{
    const int start_iter = iter_;
    const int stop_iter = iter_ + iters;
    int average_loss = this->param_.average_loss();
    losses_.clear();
    smoothed_loss_ = 0;

    // 对于每一次训练时的迭代(遍历整个网络)
    while (iter_ < stop_iter)
    {
        // zero-init the params
        net_->ClearParamDiffs();

        // test_initialization默认为true 
        if (param_.test_interval() && iter_ % param_.test_interval() == 0
        && (iter_ > 0 || param_.test_initialization())
        && Caffe::root_solver()) 
        {
            TestAll();

            if (requested_early_exit_) 
            {
                // Break out of the while loop because stop was requested while testing.
                break;
            }
        }

        for (int i = 0; i < callbacks_.size(); ++i) 
        {
            callbacks_[i]->on_start();
        }
        
        const bool display = param_.display() && iter_ % param_.display() == 0;
        net_->set_debug_info(display && param_.debug_info());
        // accumulate the loss and gradient
        Dtype loss = 0;

        // 1. 计算loss：loss = net_->ForwardBackward(bottom_vec)其中：
        for (int i = 0; i < param_.iter_size(); ++i) 
        {
            loss += net_->ForwardBackward();
        }

        // accumulate（累积） gradients over `iter_size` x `batch_size` 
        // instances。默认情况下，iter_size=1,即默认情况下，一个iteratio一个batch 
        loss /= param_.iter_size();

        // 3. 输出 losses_
        // average the loss across iterations for smoothed reporting
        UpdateSmoothedLoss(loss, start_iter, average_loss);
        if (display) {
        LOG_IF(INFO, Caffe::root_solver()) << "Iteration " << iter_
          << ", loss = " << smoothed_loss_;
        const vector<Blob<Dtype>*>& result = net_->output_blobs();
        int score_index = 0;
        for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        const string& output_name =
            net_->blob_names()[net_->output_blob_indices()[j]];
        const Dtype loss_weight =
            net_->blob_loss_weights()[net_->output_blob_indices()[j]];
        for (int k = 0; k < result[j]->count(); ++k) {
          ostringstream loss_msg_stream;
          if (loss_weight) {
            loss_msg_stream << " (* " << loss_weight
                            << " = " << loss_weight * result_vec[k] << " loss)";
          }
          LOG_IF(INFO, Caffe::root_solver()) << "    Train net output #"
              << score_index++ << ": " << output_name << " = "
              << result_vec[k] << loss_msg_stream.str();
        }
        }
        }
        for (int i = 0; i < callbacks_.size(); ++i) {
        callbacks_[i]->on_gradients_ready();
        }

        // 2. 调用ComputeUpdateValue函数:ComputeUpdateValue() 
        ApplyUpdate();

        // Increment the internal iter_ counter -- its value should always indicate
        // the number of times the weights have been updated.
        ++iter_;

        SolverAction::Enum request = GetRequestedAction();

        // 5. 达到snapshot时调用snapshot() 
        // Save a snapshot if needed.
        if ((param_.snapshot()
         && iter_ % param_.snapshot() == 0
         && Caffe::root_solver()) ||
         (request == SolverAction::SNAPSHOT)) {
        Snapshot();
        }
        if (SolverAction::STOP == request) {
        requested_early_exit_ = true;
        // Break out of training loop.
        break;
        }
    }
}

// 功能：训练网络 
/*
对整个网络进行训练（也就是你运行Caffe训练某个模型）的时候，实际上是在运行caffe.cpp
中的train( )函数，而这个函数实际上是实例化一个Solver对象，初始化后调用了Solver中
的Solve( )方法
调用此方法训练网络，其中会调用Step()方法来迭代，迭代 param_.max_iter() - iter_ 次  
*/
template <typename Dtype>
void Solver<Dtype>::Solve(const char* resume_file) 
{
    CHECK(Caffe::root_solver());
    LOG(INFO) << "Solving " << net_->name();
    LOG(INFO) << "Learning Rate Policy: " << param_.lr_policy();

    // Initialize to false every time we start solving.
    requested_early_exit_ = false;

    if (resume_file) 
    {
        LOG(INFO) << "Restoring previous solver status from " << resume_file;
        Restore(resume_file);
    }

    // For a network that is trained by the solver, no bottom or top vecs
    // should be given, and we will just provide dummy vecs.
    int start_iter = iter_;
    Step(param_.max_iter() - iter_);
    
    // If we haven't already, save a snapshot after optimization, unless
    // overridden by setting snapshot_after_train := false
    if (param_.snapshot_after_train()
     && (!param_.snapshot() || iter_ % param_.snapshot() != 0)) 
    {
        Snapshot();
    }
    
    if (requested_early_exit_) 
    {
        LOG(INFO) << "Optimization stopped early.";
        return;
    }
    
    // After the optimization is done, run an additional train and test pass to
    // display the train and test loss/outputs if appropriate (based on the
    // display and test_interval settings, respectively).  Unlike in the rest of
    // training, for the train net we only run a forward pass as we've already
    // updated the parameters "max_iter" times -- this final pass is only done to
    // display the loss, which is computed in the forward pass.
    if (param_.display() && iter_ % param_.display() == 0) 
    {
        int average_loss = this->param_.average_loss();
        Dtype loss;
        net_->Forward(&loss);

        UpdateSmoothedLoss(loss, start_iter, average_loss);

        LOG(INFO) << "Iteration " << iter_ << ", loss = " << smoothed_loss_;
    }

    // 4. 达到test_interval时调用Test() 
    if (param_.test_interval() && iter_ % param_.test_interval() == 0) {
    TestAll();
    }
    LOG(INFO) << "Optimization Done.";
}

template <typename Dtype>
void Solver<Dtype>::TestAll() {
  for (int test_net_id = 0;
       test_net_id < test_nets_.size() && !requested_early_exit_;
       ++test_net_id) {
    Test(test_net_id);
  }
}

template <typename Dtype>
void Solver<Dtype>::Test(const int test_net_id) 
{
    CHECK(Caffe::root_solver());
    LOG(INFO) << "Iteration " << iter_
    << ", Testing net (#" << test_net_id << ")";

    // 检查是否有layer共享于多个网络
    CHECK_NOTNULL(test_nets_[test_net_id].get())->
    ShareTrainedLayersWith(net_.get());
    vector<Dtype> test_score;
    vector<int> test_score_output_id;

    // 1. 设置当前阶段（TRAIN还是TEST/TRAIN）

    // 2. 将test_net_指向net_,即对同一个网络操作 
    const shared_ptr<Net<Dtype> >& test_net = test_nets_[test_net_id];
    Dtype loss = 0;

    // 3. 对于每一次测试时的迭代：for (int i = 0; i < param_.test_iter(); ++i)
    for (int i = 0; i < param_.test_iter(test_net_id); ++i) 
    {
        SolverAction::Enum request = GetRequestedAction();

        // Check to see if stoppage of testing/training has been requested.
        while (request != SolverAction::NONE) 
        {
            if (SolverAction::SNAPSHOT == request) 
            {
                Snapshot();
            } 
            else if (SolverAction::STOP == request) 
            {
                requested_early_exit_ = true;
            }
            
            request = GetRequestedAction();
        }
        
        if (requested_early_exit_) 
        {
            // break out of test loop.
            break;
        }

        Dtype iter_loss;

        // ① 用下面语句给result赋值net_output_blobs_ 
        // result是所有的输出层blob 同时得到这次测试的iter_loss 
        const vector<Blob<Dtype>*>& result =
        test_net->Forward(&iter_loss);

        if (param_.test_compute_loss()) 
        {
            loss += iter_loss;
        }

        // ② 第一次测试时： 
        if (i == 0) 
        {
            for (int j = 0; j < result.size(); ++j) 
            {
                // 取每一个输出层的blob result_vec = result[j]->cpu_data()
                const Dtype* result_vec = result[j]->cpu_data();

                // 把每一个blob的数据（降为一维）存入一个vector–“test_score”
                for (int k = 0; k < result[j]->count(); ++k) 
                {
                    test_score.push_back(result_vec[k]);
                    test_score_output_id.push_back(j);
                }
            }
        } 
        else // ③ 不是第一次测试： 
        {
            int idx = 0;

            for (int j = 0; j < result.size(); ++j) 
            {
                const Dtype* result_vec = result[j]->cpu_data();

                for (int k = 0; k < result[j]->count(); ++k) 
                {
                    // 用 test_score[idx++] += result_vec[k] 
                    // 而不是 test_score.push_back(result_vec[k])
                    // 把输出层对应位置的blob值累加 
                    // test_score[idx++] += result_vec[k]
                    test_score[idx++] += result_vec[k];
                }
            }
        }
    }

    if (requested_early_exit_) 
    {
        LOG(INFO)     << "Test interrupted.";
        return;
    }

    // ④ 是否要输出Test loss
    if (param_.test_compute_loss()) 
    {
        loss /= param_.test_iter(test_net_id);
        LOG(INFO) << "Test loss: " << loss;
    }

    // ⑤ 是否要输出test_score
    for (int i = 0; i < test_score.size(); ++i) 
    {
        const int output_blob_index =
        test_net->output_blob_indices()[test_score_output_id[i]];
        const string& output_name = test_net->blob_names()[output_blob_index];
        const Dtype loss_weight = test_net->blob_loss_weights()[output_blob_index];
        ostringstream loss_msg_stream;

        // 求多次迭代Loss的平均值，也就是求多个batch的平局值，因为一次迭代用的是一个test batch-size 的图片  
        const Dtype mean_score = test_score[i] / param_.test_iter(test_net_id);

        if (loss_weight) 
        {
            loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
        }
        
        LOG(INFO) << "    Test net output #" << i << ": " << output_name << " = "
          << mean_score << loss_msg_stream.str();
    }

    // ⑥ 设置当前阶段（TRAIN还是TEST/TRAIN）
}

// 输出当前网络状态到一个文件中，不重要
template <typename Dtype>
void Solver<Dtype>::Snapshot() {
  CHECK(Caffe::root_solver());
  string model_filename;
  switch (param_.snapshot_format()) {
  case caffe::SolverParameter_SnapshotFormat_BINARYPROTO:
    model_filename = SnapshotToBinaryProto();
    break;
  case caffe::SolverParameter_SnapshotFormat_HDF5:
    model_filename = SnapshotToHDF5();
    break;
  default:
    LOG(FATAL) << "Unsupported snapshot format.";
  }

  SnapshotSolverState(model_filename);
}

template <typename Dtype>
void Solver<Dtype>::CheckSnapshotWritePermissions() {
  if (Caffe::root_solver() && param_.snapshot()) {
    CHECK(param_.has_snapshot_prefix())
        << "In solver params, snapshot is specified but snapshot_prefix is not";
    string probe_filename = SnapshotFilename(".tempfile");
    std::ofstream probe_ofs(probe_filename.c_str());
    if (probe_ofs.good()) {
      probe_ofs.close();
      std::remove(probe_filename.c_str());
    } else {
      LOG(FATAL) << "Cannot write to snapshot prefix '"
          << param_.snapshot_prefix() << "'.  Make sure "
          << "that the directory exists and is writeable.";
    }
  }
}

template <typename Dtype>
string Solver<Dtype>::SnapshotFilename(const string extension) {
  return param_.snapshot_prefix() + "_iter_" + caffe::format_int(iter_)
    + extension;
}

template <typename Dtype>
string Solver<Dtype>::SnapshotToBinaryProto() {
  string model_filename = SnapshotFilename(".caffemodel");
  LOG(INFO) << "Snapshotting to binary proto file " << model_filename;
  NetParameter net_param;
  net_->ToProto(&net_param, param_.snapshot_diff());
  WriteProtoToBinaryFile(net_param, model_filename);
  return model_filename;
}

template <typename Dtype>
string Solver<Dtype>::SnapshotToHDF5() {
  string model_filename = SnapshotFilename(".caffemodel.h5");
  LOG(INFO) << "Snapshotting to HDF5 file " << model_filename;
  net_->ToHDF5(model_filename, param_.snapshot_diff());
  return model_filename;
}
// 从一个文件中读入网络状态，并可以从那个状态恢复，不重要 
template <typename Dtype>
void Solver<Dtype>::Restore(const char* state_file) {
  CHECK(Caffe::root_solver());
  string state_filename(state_file);
  if (state_filename.size() >= 3 &&
      state_filename.compare(state_filename.size() - 3, 3, ".h5") == 0) {
    RestoreSolverStateFromHDF5(state_filename);
  } else {
    RestoreSolverStateFromBinaryProto(state_filename);
  }
}

template <typename Dtype>
void Solver<Dtype>::UpdateSmoothedLoss(Dtype loss, int start_iter,
    int average_loss) {
  if (losses_.size() < average_loss) {
    losses_.push_back(loss);
    int size = losses_.size();
    smoothed_loss_ = (smoothed_loss_ * (size - 1) + loss) / size;
  } else {
    int idx = (iter_ - start_iter) % average_loss;
    smoothed_loss_ += (loss - losses_[idx]) / average_loss;
    losses_[idx] = loss;
  }
}

INSTANTIATE_CLASS(Solver);

}  // namespace caffe
