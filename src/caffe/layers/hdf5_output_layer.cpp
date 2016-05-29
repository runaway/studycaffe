#include <vector>

#include "hdf5.h"
#include "hdf5_hl.h"

#include "caffe/layers/hdf5_output_layer.hpp"
#include "caffe/util/hdf5.hpp"

namespace caffe {

template <typename Dtype>
void HDF5OutputLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,  
    const vector<Blob<Dtype>*>& top) {  
  // �����ļ��е��ļ���  
  file_name_ = this->layer_param_.hdf5_output_param().file_name();  
  // ���ļ�  
  file_id_ = H5Fcreate(file_name_.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,  
                       H5P_DEFAULT);  
  CHECK_GE(file_id_, 0) << "Failed to open HDF5 file" << file_name_;  
  file_opened_ = true;// �����ļ��򿪱�־  
}  
  
template <typename Dtype>  
HDF5OutputLayer<Dtype>::~HDF5OutputLayer<Dtype>() {  
  if (file_opened_) {  
    herr_t status = H5Fclose(file_id_);  
    CHECK_GE(status, 0) << "Failed to close HDF5 file " << file_name_;  
  }  
}  
  
// ��blob��ŵ�hdf5�ļ�  
// ���ݺ����  
template <typename Dtype>  
void HDF5OutputLayer<Dtype>::SaveBlobs() {  
  // TODO: no limit on the number of blobs  
  LOG(INFO) << "Saving HDF5 file " << file_name_;  
  CHECK_EQ(data_blob_.num(), label_blob_.num()) <<  
      "data blob and label blob must have the same batch size";  
  hdf5_save_nd_dataset(file_id_, HDF5_DATA_DATASET_NAME, data_blob_);  
  hdf5_save_nd_dataset(file_id_, HDF5_DATA_LABEL_NAME, label_blob_);  
  LOG(INFO) << "Successfully saved " << data_blob_.num() << " rows";  
}  
  
// ʵ���Ͼ��Ǵ�bottom��������������ݴ�ŵ�hdf5�ļ�  
template <typename Dtype>  
void HDF5OutputLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,  
      const vector<Blob<Dtype>*>& top) {  
  CHECK_GE(bottom.size(), 2);  
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());  
  // �ı�data_blob_����״�Լ�label_blob_����״  
  data_blob_.Reshape(bottom[0]->num(), bottom[0]->channels(),  
                     bottom[0]->height(), bottom[0]->width());  
  label_blob_.Reshape(bottom[1]->num(), bottom[1]->channels(),  
                     bottom[1]->height(), bottom[1]->width());  
  const int data_datum_dim = bottom[0]->count() / bottom[0]->num();  
  const int label_datum_dim = bottom[1]->count() / bottom[1]->num();  
  
  // ��bottom[0]��[1]���Ƶ�data_blob_��label_blob_  
  for (int i = 0; i < bottom[0]->num(); ++i) {  
    caffe_copy(data_datum_dim, &bottom[0]->cpu_data()[i * data_datum_dim],  
        &data_blob_.mutable_cpu_data()[i * data_datum_dim]);  
    caffe_copy(label_datum_dim, &bottom[1]->cpu_data()[i * label_datum_dim],  
        &label_blob_.mutable_cpu_data()[i * label_datum_dim]);  
  }  
  // ��ŵ��ļ�  
  SaveBlobs();  
}  
  
// ������  
template <typename Dtype>  
void HDF5OutputLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,  
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {  
  return;  
}  
  
#ifdef CPU_ONLY  
STUB_GPU(HDF5OutputLayer);  
#endif  
  
INSTANTIATE_CLASS(HDF5OutputLayer);  
REGISTER_LAYER_CLASS(HDF5Output);  
  
}  // namespace caffe  

//}  // namespace caffe
