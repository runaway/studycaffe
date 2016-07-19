#ifndef CAFFE_BLOB_HPP_
#define CAFFE_BLOB_HPP_

#include <algorithm>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"

const int kMaxBlobAxes = 32;

/*
ʵ����BLOB��������������
��1��data��ǰ�򴫲����õ�������
��2��diff�����򴫲����õ�������
��3��shape������data��diff��shape����
��ôΧ�������������ж�Ӧ�ķ�����
*/

namespace caffe {

/**
 * @brief A wrapper around SyncedMemory holders serving as the basic
 *        computational unit through which Layer%s, Net%s, and Solver%s
 *        interact.
 *
 * TODO(dox): more thorough description.
 */
// BLOB��SyncedMemory�İ�����  
// ��SyncedMemory�ķ�װ��������Layer��Net��Solver֮����н���ʱʹ�õĻ������㵥Ԫ��
template <typename Dtype>
class Blob {
 public:
  Blob()
       : data_(), diff_(), count_(0), capacity_(0) {}


  /// @brief Deprecated; use <code>Blob(const vector<int>& shape)</code>.
  explicit Blob(const int num, const int channels, const int height,
      const int width);
  explicit Blob(const vector<int>& shape); // �Ƽ�ʹ�����  

  /// @brief Deprecated; use <code>Reshape(const vector<int>& shape)</code>.
  void Reshape(const int num, const int channels, const int height,
      const int width);
  /**
   * @brief Change the dimensions of the blob, allocating new memory if
   *        necessary.
   *
   * This function can be called both to create an initial allocation
   * of memory, and to adjust the dimensions of a top blob during Layer::Reshape
   * or Layer::Forward. When changing the size of blob, memory will only be
   * reallocated if sufficient memory does not already exist, and excess memory
   * will never be freed.
   *
   * Note that reshaping an input blob and immediately calling Net::Backward is
   * an error; either Net::Forward or Net::Reshape need to be called to
   * propagate the new input shape to higher layers.
   */
  void Reshape(const vector<int>& shape); // �Ƽ�ʹ�����
  void Reshape(const BlobShape& shape);
  void ReshapeLike(const Blob& other);

  // ������ݵ�ά��,�Կո�ָ���������һάά��(total)  
  // ��shape_�е������Լ�count_ƴ�ӳ��ַ��������أ���Ҫ����־��
  inline string shape_string() const {
    ostringstream stream;
    for (int i = 0; i < shape_.size(); ++i) {
      stream << shape_[i] << " ";
    }
    stream << "(" << count_ << ")";
    return stream.str();
  }

  // ��ȡblob��ά����Ϣ
  inline const vector<int>& shape() const { return shape_; }
  /**
   * @brief Returns the dimension of the index-th axis (or the negative index-th
   *        axis from the end, if index is negative).
   *
   * @param index the axis index, which may be negative as it will be
   *        "canonicalized" using CanonicalAxisIndex.
   *        Dies on out of range index.
   */
  // ����Ӹ���ά�ȵ����һ��ά�ȵ�
  // ��ȡblob��ĳһ�������ϵ�ά�ȣ�index����Ǹ��������������һ�������Ὺʼ��-index�������ᣩ��
  //       ����index����ڼ��������ᣬindex����ȡ������
  inline int shape(int index) const {
    return shape_[CanonicalAxisIndex(index)];
  }

  // �������ݵ�ά�� 
  // ��ȡblob��������ĸ�����
  inline int num_axes() const { return shape_.size(); }

  // �������ݵ�����ά�ȵ����,�����ݵĸ���  
  // ��ȡblob�����ݵĸ�����
  inline int count() const { return count_; } 

  /**
   * @brief Compute the volume of a slice; i.e., the product of dimensions
   *        among a range of axes.
   *
   * @param start_axis The first axis to include in the slice.
   *
   * @param end_axis The first axis to exclude from the slice.
   */
   //  ��ȡ��start_axis�����ᵽend_axis������֮������ݵĸ�����
  inline int count(int start_axis, int end_axis) const {
    // �ж�ά�ȵ������Ƿ��ڷ�Χ�� 
    CHECK_LE(start_axis, end_axis);
    CHECK_GE(start_axis, 0);
    CHECK_GE(end_axis, 0);
    CHECK_LE(start_axis, num_axes());
    CHECK_LE(end_axis, num_axes());
    int count = 1;
    for (int i = start_axis; i < end_axis; ++i) {
      count *= shape(i);
    }
    return count;
  }
  /**
   * @brief Compute the volume of a slice spanning from a particular first
   *        axis to the final axis.
   *
   * @param start_axis The first axis to include in the slice.
   */
  // ������ά�ȵ�����ά��֮����������ݸ��� 
  inline int count(int start_axis) const {
    return count(start_axis, num_axes());
  }

  /**
   * @brief Returns the 'canonical' version of a (usually) user-specified axis,
   *        allowing for negative indexing (e.g., -1 for the last axis).
   *
   * @param axis_index the axis index.
   *        If 0 <= index < num_axes(), return index.
   *        If -num_axes <= index <= -1, return (num_axes() - (-index)),
   *        e.g., the last axis index (num_axes() - 1) if index == -1,
   *        the second to last if index == -2, etc.
   *        Dies on out of range index.
   */
  // ֧�ָ���ά��������������ʾ�Ӻ���ǰ�����ص�����ȷ��ά���������൱�ڽ������������е�ת���� 
  //       ��ȡ�淶��������������
  //       axis_index�����Ǹ���������Ǹ������򷵻�axis_index + num_axes()����������-axis_index������
  inline int CanonicalAxisIndex(int axis_index) const {
  // �ж��Ƿ��ڷ�Χ��[-numaxes, numaxes] 
    CHECK_GE(axis_index, -num_axes())
        << "axis " << axis_index << " out of range for " << num_axes()
        << "-D Blob with shape " << shape_string();
    CHECK_LT(axis_index, num_axes())
        << "axis " << axis_index << " out of range for " << num_axes()
        << "-D Blob with shape " << shape_string();
    if (axis_index < 0) {
      return axis_index + num_axes();
    }
    return axis_index;
  }

  /// @brief Deprecated legacy shape accessor num: use shape(0) instead.
  inline int num() const { return LegacyShape(0); }
  /// @brief Deprecated legacy shape accessor channels: use shape(1) instead.
  inline int channels() const { return LegacyShape(1); }
  /// @brief Deprecated legacy shape accessor height: use shape(2) instead.
  inline int height() const { return LegacyShape(2); }
  /// @brief Deprecated legacy shape accessor width: use shape(3) instead.
  inline int width() const { return LegacyShape(3); }

  // ����ĳһ�������ϵ����ݸ�����indexΪ����������
  inline int LegacyShape(int index) const {
    // ���blob��ά�ȸ����ǲ���С��4��Ҳ����ǰ��blobֻ����ά���������ڵ�blobӦ��Ϊ��ͨ�ö������˴�����ά�ķ��� 
    CHECK_LE(num_axes(), 4)
        << "Cannot use legacy accessors on Blobs with > 4 axes.";

    // ���ά�������ǲ���С��4  
    CHECK_LT(index, 4);

    // ���ά�������ǲ��Ǵ���-4 
    CHECK_GE(index, -4);
    if (index >= num_axes() || index < -num_axes()) {
      // Axis is out of range, but still in [0, 3] (or [-4, -1] for reverse
      // indexing) -- this special case simulates the one-padding used to fill
      // extraneous axes of legacy blobs.
      return 1;
    }
    return shape(index);
  }

  // �������ݵ�ƫ��

  // ����һά����ƫ����   
  inline int offset(const int n, const int c = 0, const int h = 0,
      const int w = 0) const {
    CHECK_GE(n, 0);
    CHECK_LE(n, num());
    CHECK_GE(channels(), 0);
    CHECK_LE(c, channels());
    CHECK_GE(height(), 0);
    CHECK_LE(h, height());
    CHECK_GE(width(), 0);
    CHECK_LE(w, width());
    return ((n * channels() + c) * height() + h) * width() + w;
  }

  // ����һά����ƫ����,ֻ���������õ���vector<int>  
  inline int offset(const vector<int>& indices) const {
    CHECK_LE(indices.size(), num_axes());
    int offset = 0;
    for (int i = 0; i < num_axes(); ++i) {
      offset *= shape(i);
      if (indices.size() > i) {
        CHECK_GE(indices[i], 0);
        CHECK_LT(indices[i], shape(i));
        offset += indices[i];
      }
    }
    return offset;
  }
  /**
   * @brief Copy from a source Blob.
   *
   * @param source the Blob to copy from
   * @param copy_diff if false, copy the data; if true, copy the diff
   * @param reshape if false, require this Blob to be pre-shaped to the shape
   *        of other (and die otherwise); if true, Reshape this Blob to other's
   *        shape if necessary
   * �Ӹ�����blob���и��ƣ����copy_diff=true���µ�blob���Ƶ���diff,���reshape=true��ı���blob����״ 
   */  
  void CopyFrom(const Blob<Dtype>& source, bool copy_diff = false,  
      bool reshape = false); 

  // ��ȡĳһλ�õ�����
  // ��ȡ���ڴ��µ�����(ǰ�򴫲����õ�����)  
  inline Dtype data_at(const int n, const int c, const int h,  
      const int w) const {  
    return cpu_data()[offset(n, c, h, w)];  
  }  
  // ��ȡ���ڴ��µ�diff����(��������)  
  inline Dtype diff_at(const int n, const int c, const int h,  
      const int w) const {  
    return cpu_diff()[offset(n, c, h, w)];  
  }  
  // ��ȡ���ڴ��µ�����(ǰ�򴫲����õ�����)  
  inline Dtype data_at(const vector<int>& index) const {  
    return cpu_data()[offset(index)];  
  }  

  // ��ȡĳһλ�õĲ�ֵ
  // ��ȡ���ڴ��µ�diff����(��������)  
  inline Dtype diff_at(const vector<int>& index) const {  
    return cpu_diff()[offset(index)];  
  }  

  // ��ȡ����ָ��
  // ͬ���ڴ�shared_ptr��������share_ptr�Ŀ������аٶȣ����ü���������ƣ�  
  inline const shared_ptr<SyncedMemory>& data() const {  
    CHECK(data_);  
    return data_;  
  }  

  // ��ȡ��ֵ����ָ��
  inline const shared_ptr<SyncedMemory>& diff() const {  
    CHECK(diff_);  
    return diff_;  
  }  
  
  // ����  
  const Dtype* cpu_data() const;
  void set_cpu_data(Dtype* data);
  const int* gpu_shape() const;
  const Dtype* gpu_data() const;
  const Dtype* cpu_diff() const;
  const Dtype* gpu_diff() const;
  Dtype* mutable_cpu_data();
  Dtype* mutable_gpu_data();
  Dtype* mutable_cpu_diff();
  Dtype* mutable_gpu_diff();
  // ����Y=alpha?X+beta?Y

  // ���ܣ�����params_��blob��ֵ
  void Update();  
  // ��protobuf���л��ļ���ȡblob����  
  void FromProto(const BlobProto& proto, bool reshape = true);  
  // ���������л�Ϊprotobuf�ļ�  
  void ToProto(BlobProto* proto, bool write_diff = false) const;

  /// @brief Compute the sum of absolute values (L1 norm) of the data.
  Dtype asum_data() const;
  /// @brief Compute the sum of absolute values (L1 norm) of the diff.
  Dtype asum_diff() const;
  /// @brief Compute the sum of squares (L2 norm squared) of the data.
  Dtype sumsq_data() const;
  /// @brief Compute the sum of squares (L2 norm squared) of the diff.
  Dtype sumsq_diff() const;

  /// @brief Scale the blob data by a constant factor.
  void scale_data(Dtype scale_factor);
  /// @brief Scale the blob diff by a constant factor.
  void scale_diff(Dtype scale_factor);

  /**
   * @brief Set the data_ shared_ptr to point to the SyncedMemory holding the
   *        data_ of Blob other -- useful in Layer%s which simply perform a copy
   *        in their Forward pass.
   *
   * This deallocates the SyncedMemory holding this Blob's data_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   */
  void ShareData(const Blob& other);
  /**
   * @brief Set the diff_ shared_ptr to point to the SyncedMemory holding the
   *        diff_ of Blob other -- useful in Layer%s which simply perform a copy
   *        in their Forward pass.
   *
   * This deallocates the SyncedMemory holding this Blob's diff_, as
   * shared_ptr calls its destructor when reset with the "=" operator. 
   * �����blob��data����Ӧ��diffָ������Blob��ʵ�����ݵĹ��� 
   * ͬʱ��Ҫע���������������������Blob�����SyncedMemory���ͷţ� 
   * ��Ϊshared_ptrָ�뱻��=���õ�ʱ��ص�����Ӧ���������� 
   */  
  void ShareDiff(const Blob& other);  

  // �жϵ�ǰblob��shape��BlobProto���Ƿ���ƥ��
  // �ж���״�Ƿ����  
  bool ShapeEquals(const BlobProto& other);  
  
 protected:  
  // ǰ�򴫲�������  
  shared_ptr<SyncedMemory> data_;  
  // diff�Ƿ��򴫲�������  
  shared_ptr<SyncedMemory> diff_;  
  // �ɵ���״����  
  shared_ptr<SyncedMemory> shape_data_;  
  // �µ���״����  
  vector<int> shape_;  
  // ���ݵĸ���  
  int count_;  
  // ����  
  int capacity_;  
  
  DISABLE_COPY_AND_ASSIGN(Blob);  
};  // class Blob  
  
}  // namespace caffe  

#endif  // CAFFE_BLOB_HPP_
