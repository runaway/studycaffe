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

// LayerResistry�Ĺ��ܼܺ򵥣����ǽ���Ͷ�Ӧ���ַ������ͷ��뵽һ��map����ȥ����
// �������á���Ҫ����ע����Ĺ���
template <typename Dtype>
class LayerRegistry 
{
public:

    // ����ָ��Creator�����ص���Layer<Dtype>���͵�ָ��  
    typedef shared_ptr<Layer<Dtype> > (*Creator)(const LayerParameter&);
    
    // CreatorRegistry���ַ������Ӧ��Creator��ӳ��  
    typedef std::map<string, Creator> CreatorRegistry;  

    // typedef��http://www.kuqin.com/language/20090322/41866.html
    // CreatorΪһ������ָ�룬����������shared_ptr<Layer<Dtype> >
    static CreatorRegistry& Registry() 
    {
        static CreatorRegistry* g_registry_ = new CreatorRegistry();
        return *g_registry_;
    }
    
    // �������ͣ��Լ�����ָ�룬���뵽ע��� 
    // Adds a creator.
    static void AddCreator(const string& type, Creator creator) 
    {
        CreatorRegistry& registry = Registry();
        CHECK_EQ(registry.count(type), 0)
        << "Layer type " << type << " already registered.";
        registry[type] = creator;
    }

    // Get a layer using a LayerParameter.
    // 3-2����������ͣ�������
    static shared_ptr<Layer<Dtype> > CreateLayer(const LayerParameter& param) 
    {  
    
        if (Caffe::root_solver()) 
        {  
            LOG(INFO) << "Creating layer " << param.name();  
        }  
        
        // �Ӳ����л�������ַ���  
        const string& type = param.type();  
        
        // ���ע���ָ��  
        CreatorRegistry& registry = Registry(); 
        
        // �����Ƿ���ҵ�����type��Creator  
        CHECK_EQ(registry.count(type), 1) << "Unknown layer type: " << type  
        << " (known types: " << LayerTypeListString() << ")";  

        // �����ž�ֱ��new��һ���µ�layer�ࡣDataLayer,ConvolutionLayer,...
        // ���ö�Ӧ�Ĳ��Creator����  
        return registry[type](param);  
    }  

    // 3-3���ز�������б�
    static vector<string> LayerTypeList() 
    {  
        // ���ע���  
        CreatorRegistry& registry = Registry();  
        vector<string> layer_types;  
        
        // ����ע���ѹ��layer_types�ַ�������  
        for (typename CreatorRegistry::iterator iter = registry.begin();  
             iter != registry.end(); 
             ++iter) 
        {  
            layer_types.push_back(iter->first);  
        }  
             
        return layer_types;  
    }  

private:
    
    // ��ֹʵ��������Ϊ���඼�Ǿ�̬������������˽�е�  
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


// 3-6���⻹������һ����ע����
// LayerRegisterer  
// �Լ�������ע����  
// �Թ�����ĺ����ʹ��  
template <typename Dtype>  
class LayerRegisterer 
{  
public:  
    
    // ���ע�����Ĺ��캯��  
    LayerRegisterer(const string& type,  
                  shared_ptr<Layer<Dtype> > (*creator)(const LayerParameter&)) 
    {  
        // LOG(INFO) << "Registering layer type: " << type;  
        // ���ǵ��õĲ�ע����еļ���Creator��������ע���  
        LayerRegistry<Dtype>::AddCreator(type, creator);  
    }  
};  

/*
����������
Ϊ�˷������߻�Ū�˸������ע���Լ�д�Ĳ���
[cpp] view plain copy ��CODE�ϲ鿴����Ƭ�������ҵĴ���Ƭ
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

// ����g_creator_f_type(type, creator<Dtype>)���������� ��double��float���ͣ�  
#define REGISTER_LAYER_CREATOR(type, creator)                                  \
  static LayerRegisterer<float> g_creator_f_##type(#type, creator<float>);     \
  static LayerRegisterer<double> g_creator_d_##type(#type, creator<double>)    \

// ע���Լ�������࣬����Ϊtype��  
// �������type=bias����ô�������µĴ���  
// ����ĺ���ֱ�ӵ������Լ�����Ĺ��캯������һ�����ʵ��������  
// CreatorbiasLayer(const LayerParameter& param)  
// ����������Ϊ���Լ����ඨ����LayerRegisterer<float>���͵ľ�̬����g_creator_f_biasLayer��float���ͣ�ʵ���Ͼ��ǰ����Լ�������ַ������ͺ����ʵ���󶨵�ע���  
// static LayerRegisterer<float> g_creator_f_biasLayer(bias, CreatorbiasLayer)  
// ��������Ϊ���Լ����ඨ����LayerRegisterer<double>���͵ľ�̬����g_creator_d_biasLayer��double���ͣ�ʵ���Ͼ��ǰ����Լ�������ַ������ͺ����ʵ���󶨵�ע���  
// static LayerRegisterer<double> g_creator_d_biasLayer(bias, CreatorbiasLayer)  
#define REGISTER_LAYER_CLASS(type)                                             \
  template <typename Dtype>                                                    \
  shared_ptr<Layer<Dtype> > Creator_##type##Layer(const LayerParameter& param) \
  {                                                                            \
    return shared_ptr<Layer<Dtype> >(new type##Layer<Dtype>(param));           \
  }                                                                            \
  REGISTER_LAYER_CREATOR(type, Creator_##type##Layer)
  
  // ע�������REGISTER_LAYER_CLASS,REGISTER_LAYER_CREATOR
  // �¶����˶���LayerRegisterer
  // ->AddCreator->registry[type] = creator
  // �����Ļ�,��CreateLayer��������з���ֵ��.return registry[type](param);
  // ��:��data_layer.cpp�������ҵ�REGISTER_LAYER_CLASS(Data);
  // REGISTER_LAYER_CLASS(Data)����namespace caffe���,����static��.
  // ���Ե������һ��ʹ��namespace caffeʱ,�ͻ����REGISTER_LAYER_CLASS(Data).
  // �Լ�����������static�ĺ���/����(?)

}  // namespace caffe

#endif  // CAFFE_LAYER_FACTORY_H_
