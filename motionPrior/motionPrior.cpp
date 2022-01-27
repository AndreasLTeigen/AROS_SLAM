#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

#include "motionPrior.hpp"
#include "motionPriorMethods/groundTruth.hpp"


MotionPrior getMotionPriorMethod( std::string motion_prior_method )
{
    /* Retrieve the right keypoint matching class from the corresponding config string */

    if ( motion_prior_method == "constant" )
    {
        // Brute force KNN matching of keypoints with lowes ratio test
        return MotionPrior::CONSTANT;
    }
    else if ( motion_prior_method == "gt" )
    {
        return MotionPrior::GT;
    }
    else
    {
        std::cout << "ERROR: MOTION PRIOR METHOD NOT FOUND" << std::endl;
        return MotionPrior::NONE;
    }
}

void calculateMotionPrior( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2, MotionPrior motion_prior_method )
{
    switch(motion_prior_method)
    {
        case MotionPrior::CONSTANT:
        {
            std::cout << "ERROR: MOTION PRIOR METHOD NOT IMPLEMENTED" << std::endl;
        } break;

        case MotionPrior::GT:
        {
            cv::Mat T = motionPriorGT(frame1, frame2);
        } break;

        default:
        {
            std::cout << "ERROR: MOTION PRIOR ALGORITHM NOT IMPLEMENTED" << std::endl;
        }
    }
}