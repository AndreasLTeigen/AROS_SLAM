#ifndef extraction_gt_h
#define extraction_gt_h

#include "../../dataStructures/frameData.hpp"
#include "../../dataStructures/map3D.hpp"

void depthGTwBucketing(cv::Mat& img, std::shared_ptr<FrameData> frame, std::vector<cv::KeyPoint>& kpts, std::shared_ptr<Map3D> map_3d, int h_n_buckets, int w_n_buckets);

#endif