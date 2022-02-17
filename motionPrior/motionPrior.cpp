#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

#include "motionPrior.hpp"
#include "motionPriorMethods/groundTruth.hpp"
#include "../util/util.hpp"


std::shared_ptr<MotionPrior> getMotionPrior( std::string motion_prior_method )
{
    if ( motion_prior_method == "constant" )
    {
        //return std::make_shared<ConstantMP>();
    }
    else if ( motion_prior_method == "gt" )
    {
        return std::make_shared<GroundTruthMP>();
    }
    else
    {
        std::cerr << "ERROR: MOTION PRIOR METHOD NOT FOUND" << std::endl;
        return std::make_shared<NoneMP>();
    }
}




void NoneMP::calculate( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2 )
{
    std::cerr << "ERROR: MOTION PRIOR ALGORITHM NOT IMPLEMENTED" << std::endl;
}