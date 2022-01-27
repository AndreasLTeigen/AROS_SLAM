#ifndef motionPrior_gt_h
#define motionPrior_gt_h

#include "../../dataStructures/frameData.hpp"
#include "../../dataStructures/map3D.hpp"

cv::Mat motionPriorGT(std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2);

#endif