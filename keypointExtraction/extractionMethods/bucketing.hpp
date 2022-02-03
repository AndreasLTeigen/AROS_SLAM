#ifndef bucketing_h
#define bucketing_h

#include <vector>

#include "../../dataStructures/frameData.hpp"
#include "../../dataStructures/keypoint.hpp"

void globalNaiveBucketing(cv::Mat& img, std::shared_ptr<FrameData> frame, int h_n_buckets, int w_n_buckets);
void print2DBuckets(std::vector<std::vector<std::vector<std::shared_ptr<KeyPoint2>>>> buckets);
std::shared_ptr<KeyPoint2> getBestKeypointInBucket(std::vector<std::shared_ptr<KeyPoint2>> bucket);

void globalNaiveBucketing(cv::Mat& img, std::vector<cv::KeyPoint>& kpts, int h_n_buckets, int w_n_buckets);
void print2DBuckets(std::vector<std::vector<std::vector<cv::KeyPoint>>> buckets);
cv::KeyPoint getBestKeypointInBucket(std::vector<cv::KeyPoint> bucket);


#endif