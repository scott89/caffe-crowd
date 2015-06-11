#include <leveldb/db.h>
#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<typename Dtype>
MapDataLayer<Dtype>::~MapDataLayer<Dtype>(){
  this->JoinPrefetchThread();
}

template<typename Dtype>
TransformationParameter MapDataLayer<Dtype>::label_trans_param(
      const TransformationParameter& trans_param){
  // Initialize label_transformer and set scale to 1
  // and clear mean file
  int crop_size = trans_param.crop_size();
  bool mirror = trans_param.mirror();

  TransformationParameter label_transform_param;
  label_transform_param.set_scale(1);
  label_transform_param.set_crop_size(crop_size);
  label_transform_param.set_mirror(mirror);
  label_transform_param.clear_mean_file();
  return label_transform_param;
}

template <typename Dtype>
void MapDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Initialize DB
  db_.reset(db::GetDB(this->layer_param_.data_param().backend()));
  db_->Open(this->layer_param_.data_param().source(), db::READ);
  iter_.reset(db_->NewCursor());

  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
                        this->layer_param_.data_param().rand_skip();
    LOG(INFO) << "Skipping first" << skip << " data points.";
    while (skip-- > 0) {
      iter_->Next();
    }
  }

  // Read a data point and use it to initialize the top blob.
  BlobProtoVector maps;
  maps.ParseFromString(iter_->value());
  CHECK(maps.blobs_size() == 2) << "MapDataLayer accepts BlobProtoVector with"
                                << " 2 BlobProtos: data and label.";
  BlobProto dataMap = maps.blobs(0);
  BlobProto labelMap = maps.blobs(1);

  // do not support mirror and crop for the moment
  int crop_size = this->layer_param_.transform_param().crop_size();
  bool mirror = this->layer_param_.transform_param().mirror();
  CHECK(crop_size == 0) << "MapDataLayer does not support cropping.";
  CHECK(!mirror) << "MapDataLayer does not support mirroring";

  // reshape data map
  top[0]->Reshape(
      this->layer_param_.data_param().batch_size(), dataMap.channels(),
      dataMap.height(), dataMap.width());
  this->prefetch_data_.Reshape(this->layer_param_.data_param().batch_size(),
      dataMap.channels(), dataMap.height(), dataMap.width());
  this->transformed_data_.Reshape(1, dataMap.channels(),
      dataMap.height(), dataMap.width());
  // reshape label map
  top[1]->Reshape(
      this->layer_param_.data_param().batch_size(), labelMap.channels(),
      labelMap.height(), labelMap.width());
  this->prefetch_label_.Reshape(this->layer_param_.data_param().batch_size(),
      labelMap.channels(), labelMap.height(), labelMap.width());
  this->transformed_label_.Reshape(1, labelMap.channels(),
                                  labelMap.height(), labelMap.width());
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
}

Datum BlobProto2Datum(const BlobProto& blob){
  Datum datum;
  datum.set_channels(blob.channels());
  datum.set_height(blob.height());
  datum.set_width(blob.width());
  datum.mutable_float_data()->CopyFrom(blob.data());
  return datum;
}

template<typename Dtype>
void MapDataLayer<Dtype>::InternalThreadEntry() {
  BlobProtoVector maps;
  Datum dataMap, labelMap;
  CHECK(this->prefetch_data_.count());
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = this->prefetch_label_.mutable_cpu_data();
  const int batch_size = this->layer_param_.data_param().batch_size();

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    maps.ParseFromString(iter_->value());
    // Data transformer only accepts Datum
    dataMap = BlobProto2Datum(maps.blobs(0));
    labelMap = BlobProto2Datum(maps.blobs(1));

    // Apply data and label transformations (mirror, scale, crop...)
    int offset = this->prefetch_data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);
    this->data_transformer_->Transform(dataMap, &(this->transformed_data_));
    
    int label_offset = this->prefetch_label_.offset(item_id);
    this->transformed_label_.set_cpu_data(top_label + label_offset);
    this->label_transformer_.Transform(labelMap, &(this->transformed_label_));

    // go to the next iter
    iter_->Next();
    if (!iter_->valid()) {
      iter_->SeekToFirst();
    }
  }
}

INSTANTIATE_CLASS(MapDataLayer);
REGISTER_LAYER_CLASS(MapData);

} // caffe