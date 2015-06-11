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
class PerspectiveLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

protected:
  PerspectiveLayerTest()
      : blob_bottom_a_(new Blob<Dtype>(2, 1, 1, 1)),
        blob_bottom_b_(new Blob<Dtype>(2, 1, 1, 1)),
        blob_bottom_c_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_a_);
    filler.Fill(this->blob_bottom_b_);
    filler.Fill(this->blob_bottom_c_);
    blob_bottom_vec_.push_back(blob_bottom_a_);
    blob_bottom_vec_.push_back(blob_bottom_b_);
    blob_bottom_vec_.push_back(blob_bottom_c_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~PerspectiveLayerTest() {
    delete blob_bottom_a_;
    delete blob_bottom_b_;
    delete blob_bottom_c_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_a_;
  Blob<Dtype>* const blob_bottom_b_;
  Blob<Dtype>* const blob_bottom_c_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

};

TYPED_TEST_CASE(PerspectiveLayerTest, TestDtypesAndDevices);

TYPED_TEST(PerspectiveLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  shared_ptr<PerspectiveLayer<Dtype> > layer(new PerspectiveLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 4);
  EXPECT_EQ(this->blob_top_->width(), 5);
}

TYPED_TEST(PerspectiveLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  shared_ptr<PerspectiveLayer<Dtype> > layer(new PerspectiveLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Blob<Dtype>* top = this->blob_top_vec_[0];
  const Dtype* top_data = top->cpu_data();
  for (int num = 0; num < 2; num++) {
    Dtype slope = this->blob_bottom_a_->cpu_data()[num];
    Dtype intercept = this->blob_bottom_b_->cpu_data()[num];
    for (int row = 0; row < 4; row++) {
      for (int col = 0; col < 5; col++) {
        EXPECT_NEAR(*(top_data + top->offset(num, 0, row, col)),
                    row * slope + intercept,
                    Dtype(1e-7));
      }
    }
  }
}

TYPED_TEST(PerspectiveLayerTest, TestForwardWithMult) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  Dtype slope_mult = 0.1;
  Dtype intercept_mult = 0.2;
  layer_param.mutable_perspective_param()->set_slope_mult(slope_mult);
  layer_param.mutable_perspective_param()->set_intercept_mult(intercept_mult);
  shared_ptr<PerspectiveLayer<Dtype> > layer(new PerspectiveLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Blob<Dtype>* top = this->blob_top_vec_[0];
  const Dtype* top_data = top->cpu_data();
  for (int num = 0; num < 2; num++) {
    Dtype slope = this->blob_bottom_a_->cpu_data()[num];
    Dtype intercept = this->blob_bottom_b_->cpu_data()[num];
    for (int row = 0; row < 4; row++) {
      for (int col = 0; col < 5; col++) {
        EXPECT_NEAR(*(top_data + top->offset(num, 0, row, col)),
                    (row * slope * slope_mult + intercept * intercept_mult),
                    Dtype(1e-7));
      }
    }
  }
}

TYPED_TEST(PerspectiveLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PerspectiveLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradient(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(PerspectiveLayerTest, TestGradientWithMult) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  Dtype slope_mult = 0.1;
  Dtype intercept_mult = 0.2;
  layer_param.mutable_perspective_param()->set_slope_mult(slope_mult);
  layer_param.mutable_perspective_param()->set_intercept_mult(intercept_mult);
  PerspectiveLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradient(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}   // namespace caffe