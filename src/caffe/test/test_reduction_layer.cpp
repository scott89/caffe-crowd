#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class ReductionLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ReductionLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~ReductionLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  void TestForward(ReductionParameter_ReductionOp op, float coeff = 1) {
    LayerParameter layer_param;
    ReductionParameter* reduction_param = layer_param.mutable_reduction_param();
    reduction_param->set_operation(op);
    if (coeff != 1.0) {
      reduction_param->set_coeff(2.3);
    }
    shared_ptr<ReductionLayer<Dtype> > layer(
        new ReductionLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const int count = this->blob_bottom_->count();
    const Dtype computed_result = this->blob_top_->cpu_data()[0];
    const Dtype* in_data = this->blob_bottom_->cpu_data();
    Dtype expected_result = 0;
    for (int i = 0; i < count; ++i) {
      switch (op) {
        case ReductionParameter_ReductionOp_SUM:
          expected_result += in_data[i];
          break;
        case ReductionParameter_ReductionOp_MEAN:
          expected_result += in_data[i] / count;
          break;
        case ReductionParameter_ReductionOp_ASUM:
          expected_result += fabs(in_data[i]);
          break;
        case ReductionParameter_ReductionOp_SUM_OF_SQUARES:
          expected_result += in_data[i] * in_data[i];
          break;
        default:
          LOG(FATAL) << "Unknown reduction op: "
              << ReductionParameter_ReductionOp_Name(op);
      }
    }
    expected_result *= coeff;
    EXPECT_FLOAT_EQ(expected_result, computed_result)
        << "Incorrect result computed with op "
        << ReductionParameter_ReductionOp_Name(op) << ", coeff " << coeff;
  }

  void TestGradient(ReductionParameter_ReductionOp op, float coeff = 1) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    ReductionParameter* reduction_param = layer_param.mutable_reduction_param();
    reduction_param->set_operation(op);
    reduction_param->set_coeff(coeff);
    ReductionLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 2e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ReductionLayerTest, TestDtypesAndDevices);

TYPED_TEST(ReductionLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  shared_ptr<ReductionLayer<Dtype> > layer(
      new ReductionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(ReductionLayerTest, TestSum) {
  const ReductionParameter_ReductionOp kOp = ReductionParameter_ReductionOp_SUM;
  this->TestForward(kOp);
}

TYPED_TEST(ReductionLayerTest, TestSumCoeff) {
  const ReductionParameter_ReductionOp kOp = ReductionParameter_ReductionOp_SUM;
  const float kCoeff = 2.3;
  this->TestForward(kOp, kCoeff);
}

TYPED_TEST(ReductionLayerTest, TestSumGradient) {
  const ReductionParameter_ReductionOp kOp = ReductionParameter_ReductionOp_SUM;
  this->TestGradient(kOp);
}

TYPED_TEST(ReductionLayerTest, TestSumCoeffGradient) {
  const ReductionParameter_ReductionOp kOp = ReductionParameter_ReductionOp_SUM;
  const float kCoeff = 2.3;
  this->TestGradient(kOp, kCoeff);
}

TYPED_TEST(ReductionLayerTest, TestMean) {
  const ReductionParameter_ReductionOp kOp =
      ReductionParameter_ReductionOp_MEAN;
  this->TestForward(kOp);
}

TYPED_TEST(ReductionLayerTest, TestMeanCoeff) {
  const ReductionParameter_ReductionOp kOp =
      ReductionParameter_ReductionOp_MEAN;
  const float kCoeff = 2.3;
  this->TestForward(kOp, kCoeff);
}

TYPED_TEST(ReductionLayerTest, TestMeanGradient) {
  const ReductionParameter_ReductionOp kOp =
      ReductionParameter_ReductionOp_MEAN;
  this->TestGradient(kOp);
}

TYPED_TEST(ReductionLayerTest, TestMeanCoeffGradient) {
  const ReductionParameter_ReductionOp kOp =
      ReductionParameter_ReductionOp_MEAN;
  const float kCoeff = 2.3;
  this->TestGradient(kOp, kCoeff);
}

TYPED_TEST(ReductionLayerTest, TestAbsSum) {
  const ReductionParameter_ReductionOp kOp =
      ReductionParameter_ReductionOp_ASUM;
  this->TestForward(kOp);
}

TYPED_TEST(ReductionLayerTest, TestAbsSumCoeff) {
  const ReductionParameter_ReductionOp kOp =
      ReductionParameter_ReductionOp_ASUM;
  const float kCoeff = 2.3;
  this->TestForward(kOp, kCoeff);
}

TYPED_TEST(ReductionLayerTest, TestAbsSumGradient) {
  const ReductionParameter_ReductionOp kOp =
      ReductionParameter_ReductionOp_ASUM;
  this->TestGradient(kOp);
}

TYPED_TEST(ReductionLayerTest, TestAbsSumCoeffGradient) {
  const ReductionParameter_ReductionOp kOp =
      ReductionParameter_ReductionOp_ASUM;
  const float kCoeff = 2.3;
  this->TestGradient(kOp, kCoeff);
}

TYPED_TEST(ReductionLayerTest, TestSumOfSquares) {
  const ReductionParameter_ReductionOp kOp =
      ReductionParameter_ReductionOp_SUM_OF_SQUARES;
  this->TestForward(kOp);
}

TYPED_TEST(ReductionLayerTest, TestSumOfSquaresCoeff) {
  const ReductionParameter_ReductionOp kOp =
      ReductionParameter_ReductionOp_SUM_OF_SQUARES;
  const float kCoeff = 2.3;
  this->TestForward(kOp, kCoeff);
}

TYPED_TEST(ReductionLayerTest, TestSumOfSquaresGradient) {
  const ReductionParameter_ReductionOp kOp =
      ReductionParameter_ReductionOp_SUM_OF_SQUARES;
  this->TestGradient(kOp);
}

TYPED_TEST(ReductionLayerTest, TestSumOfSquaresCoeffGradient) {
  const ReductionParameter_ReductionOp kOp =
      ReductionParameter_ReductionOp_SUM_OF_SQUARES;
  const float kCoeff = 2.3;
  this->TestGradient(kOp, kCoeff);
}

}  // namespace caffe
