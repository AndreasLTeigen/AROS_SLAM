#include <opencv2/opencv.hpp>

#include "copyMotionPrior.hpp"



std::shared_ptr<Pose> CopyMPPC::calculate( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2 )
{
    return frame1->getRelPose( frame2 );
}