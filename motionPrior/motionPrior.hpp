#ifndef motionPrior_h
#define motionPrior_h

#include "../dataStructures/frameData.hpp"

enum class MotionPrior {CONSTANT, GT, NONE};

MotionPrior getMotionPriorMethod( std::string motion_prior_method );
void calculateMotionPrior( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2, MotionPrior motion_prior_method );

#endif