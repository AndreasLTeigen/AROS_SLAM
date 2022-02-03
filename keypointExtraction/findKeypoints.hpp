#ifndef findKeypoints_h
#define findKeypoints_h

#include <opencv2/opencv.hpp>

#include "../dataStructures/frameData.hpp"
#include "../dataStructures/map3D.hpp"

enum class Detector {ORB, ORB_NB, ORB_NB_GT, NONE};
enum class Descriptor {ORB, NONE};

Detector getDetectionMethod( std::string detect_method );
Descriptor getDescriptionMethod( std::string desc_method );
void findKeypoints(cv::Mat& img, std::shared_ptr<FrameData> frame, std::shared_ptr<Map3D> map_3d, Detector detect_type, Descriptor desc_type);

#endif