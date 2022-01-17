#ifndef P5OR_h
#define P5OR_h

#include "../../dataStructures/frameData.hpp"
#include "../../dataStructures/pose.hpp"

std::shared_ptr<Pose> do5pointAlgOutlierRejection(std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2, cv::Mat K_matrix);

#endif