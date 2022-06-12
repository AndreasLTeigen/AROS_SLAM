#include <opencv2/opencv.hpp>

#include "copyMotionPrior.hpp"



std::shared_ptr<Pose> CopyMPPC::calculate( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2, cv::Mat& img )
{
    return frame1->getRelPose( frame2 );
}

void CopyMPPC::analysis( cv::Mat &img_disp, std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2 )
{
    std::cerr << "ERROR: POSE CALCULATION ANALYSIS ALGORITHM NOT IMPLEMENTED" << std::endl;
}