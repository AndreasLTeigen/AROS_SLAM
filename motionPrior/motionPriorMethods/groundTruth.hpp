#ifndef motionPrior_gt_h
#define motionPrior_gt_h

#include "../../dataStructures/frameData.hpp"
#include "../../dataStructures/map3D.hpp"

cv::Mat globalMotionPriorGT( std::shared_ptr<FrameData> frame1 );

#endif