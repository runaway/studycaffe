#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/sgd_solvers.hpp"
#include "caffe/solver.hpp"

#include "caffe/test/test_caffe_main.hpp"

using std::ostringstream;

namespace caffe {

template <typename TypeParam>
class SolverTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  virtual void InitSolverFromProtoString(const string& proto) {
    SolverParameter param;
    CHECK(google::protobuf::TextFormat::ParseFromString(proto, &param));
    // Set the solver_mode according to current Caffe::mode.
    switch (Caffe::mode()) {
      case Caffe::CPU:
        param.set_solver_mode(SolverParameter_SolverMode_CPU);
        break;
      case Caffe::GPU:
        param.set_solver_mode(SolverParameter_SolverMode_GPU);
        break;
      default:
        LOG(FATAL) << "Unknown Caffe mode: " << Caffe::mode();
    }
    solver_.reset(new SGDSolver<Dtype>(param));
  }

  shared_ptr<Solver<Dtype> > solver_;
};

TYPED_TEST_CASE(SolverTest, TestDtypesAndDevices);
/*
test_interval:interval是区间的意思，所有该参数表示：训练的时候，每迭代500次就进行一次测试。
caffe在训练的过程是边训练边测试的。训练过程中每500次迭代（也就是32000个训练样本参与了计算，batchsize为64），计算一次测试误差。计算一次测试误差就需要包含所有的测试图片（这里为10000），这样可以认为在一个epoch里，训练集中的所有样本都遍历以一遍，但测试集的所有样本至少要遍历一次，至于具体要多少次，也许不是整数次，这就要看代码，大致了解下这个过程就可以了。
*/
TYPED_TEST(SolverTest, TestInitTrainTestNets) {
  const string& proto =
     "test_interval: 10 "
     "test_iter: 10 "
     "test_state: { stage: 'with-softmax' }"
     "test_iter: 10 "
     "test_state: {}"
     "net_param { "
     "  name: 'TestNetwork' "
     "  layer { "
     "    name: 'data' "
     "    type: 'DummyData' "
     "    dummy_data_param { "
     "      shape { "
     "        dim: 5 "
     "        dim: 2 "
     "        dim: 3 "
     "        dim: 4 "
     "      } "
     "      shape { "
     "        dim: 5 "
     "      } "
     "    } "
     "    top: 'data' "
     "    top: 'label' "
     "  } "
     "  layer { "
     "    name: 'innerprod' "
     "    type: 'InnerProduct' "
     "    inner_product_param { "
     "      num_output: 10 "
     "    } "
     "    bottom: 'data' "
     "    top: 'innerprod' "
     "  } "
     "  layer { "
     "    name: 'accuracy' "
     "    type: 'Accuracy' "
     "    bottom: 'innerprod' "
     "    bottom: 'label' "
     "    top: 'accuracy' "
     "    exclude: { phase: TRAIN } "
     "  } "
     "  layer { "
     "    name: 'loss' "
     "    type: 'SoftmaxWithLoss' "
     "    bottom: 'innerprod' "
     "    bottom: 'label' "
     "    include: { phase: TRAIN } "
     "    include: { phase: TEST stage: 'with-softmax' } "
     "  } "
     "} ";
  this->InitSolverFromProtoString(proto);
  ASSERT_TRUE(this->solver_->net() != NULL);
  EXPECT_TRUE(this->solver_->net()->has_layer("loss"));
  EXPECT_FALSE(this->solver_->net()->has_layer("accuracy"));
  ASSERT_EQ(2, this->solver_->test_nets().size());
  EXPECT_TRUE(this->solver_->test_nets()[0]->has_layer("loss"));
  EXPECT_TRUE(this->solver_->test_nets()[0]->has_layer("accuracy"));
  EXPECT_FALSE(this->solver_->test_nets()[1]->has_layer("loss"));
  EXPECT_TRUE(this->solver_->test_nets()[1]->has_layer("accuracy"));
}

}  // namespace caffe
