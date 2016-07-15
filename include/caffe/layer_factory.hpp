/**
 * @brief A layer factory that allows one to register layers.
 * During runtime, registered layers could be called by passing a LayerParameter
 * protobuffer to the CreateLayer function:
 *
 *     LayerRegistry<Dtype>::CreateLayer(param);
 *
 * There are two ways to register a layer. Assuming that we have a layer like:
 *
 *   template <typename Dtype>
 *   class MyAwesomeLayer : public Layer<Dtype> {
 *     // your implementations
 *   };
 *
 * and its type is its C++ class name, but without the "Layer" at the end
 * ("MyAwesomeLayer" -> "MyAwesome").
 *
 * If the layer is going to be created simply by its constructor, in your c++
 * file, add the following line:
 *
 *    REGISTER_LAYER_CLASS(MyAwesome);
 *
 * Or, if the layer is going to be created by another creator function, in the
 * format of:
 *
 *    template <typename Dtype>
 *    Layer<Dtype*> GetMyAwesomeLayer(const LayerParameter& param) {
 *      // your implementation
 *    }
 *
 * (for example, when your layer has multiple backends, see GetConvolutionLayer
 * for a use case), then you can register the creator function instead, like
 *
 * REGISTER_LAYER_CREATOR(MyAwesome, GetMyAwesomeLayer)
 *
 * Note that each layer type should only be registered once.
 */

#ifndef CAFFE_LAYER_FACTORY_H_
#define CAFFE_LAYER_FACTORY_H_

#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class Layer;

// LayerResistry的功能很简单，就是将类和对应的字符串类型放入到一个map当中去，以
// 便灵活调用。主要就是注册类的功能
template <typename Dtype>
class LayerRegistry 
{
public:

    // 函数指针Creator，返回的是Layer<Dtype>类型的指针  
    typedef shared_ptr<Layer<Dtype> > (*Creator)(const LayerParameter&);
    
    // CreatorRegistry是字符串与对应的Creator的映射  
    typedef std::map<string, Creator> CreatorRegistry;  

    // typedef见http://www.kuqin.com/language/20090322/41866.html
    // Creator为一个函数指针，返回类型是shared_ptr<Layer<Dtype> >
    static CreatorRegistry& Registry() 
    {
        static CreatorRegistry* g_registry_ = new CreatorRegistry();
        return *g_registry_;
    }
    
    // 给定类型，以及函数指针，加入到注册表 
    // Adds a creator.
    static void AddCreator(const string& type, Creator creator) 
    {
        CreatorRegistry& registry = Registry();
        CHECK_EQ(registry.count(type), 0)
        << "Layer type " << type << " already registered.";
        registry[type] = creator;
    }

    // Get a layer using a LayerParameter.
    // 3-2给定层的类型，创建层
    static shared_ptr<Layer<Dtype> > CreateLayer(const LayerParameter& param) 
    {  
    
        if (Caffe::root_solver()) 
        {  
            LOG(INFO) << "Creating layer " << param.name();  
        }  
        
        // 从参数中获得类型字符串  
        const string& type = param.type();  
        
        // 获得注册表指针  
        CreatorRegistry& registry = Registry(); 
        
        // 测试是否查找到给定type的Creator  
        CHECK_EQ(registry.count(type), 1) << "Unknown layer type: " << type  
        << " (known types: " << LayerTypeListString() << ")";  

        // 这里大概就直接new了一个新的layer类。DataLayer,ConvolutionLayer,...
        // 调用对应的层的Creator函数  
        return registry[type](param);  
    }  

    // 3-3返回层的类型列表
    static vector<string> LayerTypeList() 
    {  
        // 获得注册表  
        CreatorRegistry& registry = Registry();  
        vector<string> layer_types;  
        
        // 遍历注册表压入layer_types字符串容器  
        for (typename CreatorRegistry::iterator iter = registry.begin();  
             iter != registry.end(); 
             ++iter) 
        {  
            layer_types.push_back(iter->first);  
        }  
             
        return layer_types;  
    }  

private:
    
    // 禁止实例化，因为该类都是静态函数，所以是私有的  
    // Layer registry should never be instantiated - everything is done with its
    // static variables.
    LayerRegistry() {}

    static string LayerTypeListString() 
    {
        vector<string> layer_types = LayerTypeList();
        string layer_types_str;
        
        for (vector<string>::iterator iter = layer_types.begin();
             iter != layer_types.end(); 
             ++iter) 
        {
            if (iter != layer_types.begin()) 
            {
                layer_types_str += ", ";
            }
            
            layer_types_str += *iter;
        }
         
        return layer_types_str;
    }
};


// 3-6此外还定义了一个层注册器
// LayerRegisterer  
// 自己定义层的注册器  
// 以供后面的宏进行使用  
template <typename Dtype>  
class LayerRegisterer 
{  
public:  
    
    // 层的注册器的构造函数  
    LayerRegisterer(const string& type,  
                  shared_ptr<Layer<Dtype> > (*creator)(const LayerParameter&)) 
    {  
        // LOG(INFO) << "Registering layer type: " << type;  
        // 还是调用的层注册表中的加入Creator函数加入注册表  
        LayerRegistry<Dtype>::AddCreator(type, creator);  
    }  
};  

/*
三、其他：
为了方便作者还弄了个宏便于注册自己写的层类
[cpp] view plain copy 在CODE上查看代码片派生到我的代码片
#define REGISTER_LAYER_CREATOR(type, creator)                                  \  
  static LayerRegisterer<float> g_creator_f_##type(#type, creator<float>);     \  
  static LayerRegisterer<double> g_creator_d_##type(#type, creator<double>)    \  
#define REGISTER_LAYER_CLASS(type)                                             \  
  template <typename Dtype>                                                    \  
  shared_ptr<Layer<Dtype> > Creator_##type##Layer(const LayerParameter& param) \  
  {                                                                            \  
    return shared_ptr<Layer<Dtype> >(new type##Layer<Dtype>(param));           \  
  }                                                                            \  
  REGISTER_LAYER_CREATOR(type, Creator_##type##Layer)  
*/

// 生成g_creator_f_type(type, creator<Dtype>)的两个函数 （double和float类型）  
#define REGISTER_LAYER_CREATOR(type, creator)                                  \
  static LayerRegisterer<float> g_creator_f_##type(#type, creator<float>);     \
  static LayerRegisterer<double> g_creator_d_##type(#type, creator<double>)    \

// 注册自己定义的类，类名为type，  
// 假设比如type=bias，那么生成如下的代码  
// 下面的函数直接调用你自己的类的构造函数生成一个类的实例并返回  
// CreatorbiasLayer(const LayerParameter& param)  
// 下面的语句是为你自己的类定义了LayerRegisterer<float>类型的静态变量g_creator_f_biasLayer（float类型，实际上就是把你自己的类的字符串类型和类的实例绑定到注册表）  
// static LayerRegisterer<float> g_creator_f_biasLayer(bias, CreatorbiasLayer)  
// 下面的语句为你自己的类定义了LayerRegisterer<double>类型的静态变量g_creator_d_biasLayer（double类型，实际上就是把你自己的类的字符串类型和类的实例绑定到注册表）  
// static LayerRegisterer<double> g_creator_d_biasLayer(bias, CreatorbiasLayer)  
#define REGISTER_LAYER_CLASS(type)                                             \
  template <typename Dtype>                                                    \
  shared_ptr<Layer<Dtype> > Creator_##type##Layer(const LayerParameter& param) \
  {                                                                            \
    return shared_ptr<Layer<Dtype> >(new type##Layer<Dtype>(param));           \
  }                                                                            \
  REGISTER_LAYER_CREATOR(type, Creator_##type##Layer)
  
  // 注意这里的REGISTER_LAYER_CLASS,REGISTER_LAYER_CREATOR
  // 新定义了对象LayerRegisterer
  // ->AddCreator->registry[type] = creator
  // 这样的话,在CreateLayer函数里就有返回值了.return registry[type](param);
  // 如:在data_layer.cpp最后可以找到REGISTER_LAYER_CLASS(Data);
  // REGISTER_LAYER_CLASS(Data)是在namespace caffe里的,又是static的.
  // 所以当程序第一次使用namespace caffe时,就会调用REGISTER_LAYER_CLASS(Data).
  // 以及所有其他的static的函数/对象(?)

}  // namespace caffe

#endif  // CAFFE_LAYER_FACTORY_H_
