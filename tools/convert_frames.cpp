// This program converts a set of images to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_frames [FLAGS] SUBFRAME_DIR/ SUBSEGM_DIR/ SAVE_DB MEAN_PROTO
//
//   ....

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "boost/filesystem.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;
using boost::filesystem::directory_iterator;
using boost::filesystem::path;

DEFINE_bool(shuffle, false,
            "Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "leveldb",
              "The backend {lmdb, leveldb} for storing the result");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");

void CVMatToBlobProto(const cv::Mat& cv_img, BlobProto* blob) {
  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";
  blob->set_channels(cv_img.channels());
  blob->set_height(cv_img.rows);
  blob->set_width(cv_img.cols);
  blob->set_num(1);
  blob->clear_data();
  int blob_channels = blob->channels();
  int blob_height = blob->height();
  int blob_width = blob->width();
  int blob_size = blob_channels * blob_height * blob_width;
  blob->mutable_data()->Resize(blob_size, 0.);
  for (int h = 0; h < blob_height; ++h) {
    const uchar* ptr = cv_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < blob_width; ++w) {
      for (int c = 0; c < blob_channels; ++c) {
        int blob_index = (c * blob_height + h) * blob_width + w;
        blob->set_data(blob_index, static_cast<float>(ptr[img_index++]));
      }
    }
  }
}

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of images to the leveldb/lmdb\n"
                          "format used as input for Caffe.\n"
                          "Usage:\n"
                          "    convert_frames [FLAGS] SUBFRAME_DIR/ SUBSEGM_DIR/"
                          "       SAVE_DB\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_frames");
    return 1;
  }

  path subframe_dir(argv[1]);
  path subsegm_dir(argv[2]);
  CHECK(exists(subframe_dir)) << subframe_dir << " does not exists.";
  CHECK(exists(subsegm_dir)) << subsegm_dir << " does not exists.";

  std::vector<std::pair<path, path> > samples;
  typedef std::vector<path> vec;
  vec subframes;
  // get all subframe paths
  copy(directory_iterator(subframe_dir),
       directory_iterator(),
       back_inserter(subframes));
  sort(subframes.begin(), subframes.end());
  for (vec::const_iterator it(subframes.begin()); it != subframes.end(); ++it) {
    samples.push_back(std::make_pair(*it, subsegm_dir / it->filename()));
  }
  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    shuffle(samples.begin(), samples.end());
  }
  LOG(INFO) << "A total of " << samples.size() << " images.";

  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);

  // Create new DB
  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(argv[3], db::WRITE);
  scoped_ptr<db::Transaction> txn(db->NewTransaction());

  // Storing to db
  int count = 0;

  for (int sample_id = 0; sample_id < samples.size(); ++sample_id) {
    std::string key = samples[sample_id].first.stem().string();
    BlobProtoVector sample;
    BlobProto* data_blob = sample.add_blobs();
    BlobProto* label_blob = sample.add_blobs();
    cv::Mat cv_img_bgr = ReadImageToCVMat(samples[sample_id].first.string(),
                                      resize_height, resize_width, true);
    cv::Mat cv_img_rgb;
    cv::cvtColor(cv_img_bgr, cv_img_rgb, CV_RGB2BGR);

    cv::Mat cv_segm = ReadImageToCVMat(samples[sample_id].second.string(),
                                       resize_height, resize_width, false);
    CVMatToBlobProto(cv_img_rgb, data_blob);
    CVMatToBlobProto(cv_segm, label_blob);

    // Put in db
    string out;
    CHECK(sample.SerializeToString(&out));
    txn->Put(key, out);

    if (++count % 1000 == 0) {
      // Commit db
      txn->Commit();
      txn.reset(db->NewTransaction());
      LOG(ERROR) << "Processed " << count << " files.";
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    txn->Commit();
    LOG(ERROR) << "Processed " << count << " files.";
  }

  return 0;
}
