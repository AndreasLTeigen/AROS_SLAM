#ifndef matchKeypoints_h
#define matchKeypoints_h

#include "../dataStructures/frameData.hpp"

enum class MotionPrior {constant, NONE};

MotionPrior getMotionPriorMethodMethod( std::string motion_prior_method );
void calculateMotionPrior( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2, MotionPrior motion_prior_method );

#endif