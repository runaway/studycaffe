#include <algorithm>
#include <vector>

#include "caffe/layers/hinge_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

/*
输入： 
bottom[0]: NxKx1x1维，N为样本个数，K为类别数。是预测值。 
bottom[1]: Nx1x1x1维， N为样本个数，类别为K时，每个元素的取值范围为[0,1,2,…,K-1]。是groundTruth。

输出： 
top[0]: 1x1x1x1维， 求得是hingeLoss。

关于HingeLoss： 
这里写图片描述 
p: 范数，默认是L1范数，可以在配置中设置为L1或者L2范数。 
这里写图片描述：指示函数，如果第n个样本的真实label为k，则为，否则为-1。 
tnk: bottom[0]中第n个样本，第k维的预测值。
*/
namespace caffe {

template <typename Dtype>
void HingeLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) 
{
    // 得到num个样本的dim个预测值
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

    // 得到num个样本的groundTruth
    const Dtype* label = bottom[1]->cpu_data();
    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;

    caffe_copy(count, bottom_data, bottom_diff);
    
    for (int i = 0; i < num; ++i) 
    {
        // label[i]中存储了第i个样本的真实class，取值范围[0,1,2,...,K-1]
        // 此处将第i个样本的K维预测值的label[i]处乘以-1相当于计算
        // caffe中HingeLossLayer层原理以及源码分析
        bottom_diff[i * dim + static_cast<int>(label[i])] *= -1;
    }
    
    for (int i = 0; i < num; ++i) 
    {
        for (int j = 0; j < dim; ++j) 
        {
            // 计算 caffe中HingeLossLayer层原理以及源码分析，存入 bottom_diff，
            // 即bottom[0]->mutable_cpu_diff()中
            bottom_diff[i * dim + j] = std::max(
            Dtype(0), 1 + bottom_diff[i * dim + j]);
        }
    }
    
    Dtype* loss = top[0]->mutable_cpu_data();
    
    switch (this->layer_param_.hinge_loss_param().norm()) 
    {
    // L1范数
    case HingeLossParameter_Norm_L1:

        loss[0] = caffe_cpu_asum(count, bottom_diff) / num;

    break;

    // L2范数
    case HingeLossParameter_Norm_L2:

        loss[0] = caffe_cpu_dot(count, bottom_diff, bottom_diff) / num;

    break;

    default:
        LOG(FATAL) << "Unknown Norm";
    }
}

/*
http://www.itnose.net/detail/6305786.html
反向传播原理： 
由于bottom[1]是groundtruth，不需要反传，只需要对bottom[0]进行反传，反传是损失E对t的偏导。 
以L2范数为例，求偏导为： 
这里写图片描述 
caffe中HingeLossLayer层原理以及源码分析 
其中： 
这里写图片描述 
caffe中HingeLossLayer层原理以及源码分析
*/
template <typename Dtype>
void HingeLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
    if (propagate_down[1]) 
    {
        LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
    }

    if (propagate_down[0]) 
    {
        // 说明中提到的hinge
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        const Dtype* label = bottom[1]->cpu_data();
        int num = bottom[0]->num();
        int count = bottom[0]->count();
        int dim = count / num;

        for (int i = 0; i < num; ++i) 
        {
            // 相当于求hinge*偏hinge/偏tnk部分
            bottom_diff[i * dim + static_cast<int>(label[i])] *= -1;
        }

        const Dtype loss_weight = top[0]->cpu_diff()[0];
        
        switch (this->layer_param_.hinge_loss_param().norm()) 
        {

        // L1部分反传
        case HingeLossParameter_Norm_L1:

            // L1求导的结果: 正返回1 负返回-1 0返回0
            caffe_cpu_sign(count, bottom_diff, bottom_diff);

            // scale一下
            caffe_scal(count, loss_weight / num, bottom_diff);
        break;

        // L2部分反传，就是scale一下
        case HingeLossParameter_Norm_L2:
            
            caffe_scal(count, loss_weight * 2 / num, bottom_diff);
        break;
        
        default:
            LOG(FATAL) << "Unknown Norm";
        }
    }
}

INSTANTIATE_CLASS(HingeLossLayer);
REGISTER_LAYER_CLASS(HingeLoss);

}  // namespace caffe
