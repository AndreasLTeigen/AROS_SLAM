#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

#include "motionPrior.hpp"


MotionPrior getMotionPriorMethodMethod( std::string motion_prior_method )
{
    /* Retrieve the right keypoint matching class from the corresponding config string */

    if ( motion_prior_method == "constant" )
    {
        // Brute force KNN matching of keypoints with lowes ratio test
        return MotionPrior::constant;
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
        case MotionPrior::constant:
            std::cout << "Implement constant motion prior method" << std::endl;
    }
}