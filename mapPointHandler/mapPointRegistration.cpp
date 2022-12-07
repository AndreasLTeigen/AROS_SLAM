#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "mapPointRegistration.hpp"
#include "mapPointRegistrationMethods/linearInclusiveMPReg.hpp"
#include "mapPointRegistrationMethods/depthGT.hpp"

using cv::KeyPoint;
using std::vector;
using std::shared_ptr;

std::shared_ptr<MapPointRegistrator> getMapPointRegistrator( std::string map_point_reg_method )
{
    if ( map_point_reg_method == "all" )
    {
        return std::make_shared<LinIncMPReg>();
    }
    else if ( map_point_reg_method == "depth_gt" )
    {
        return std::make_shared<depthGTMPReg>();
    }
    else
    {
        std::cerr << "Warning: Map point registration method not found." << std::endl;
        return std::make_shared<NoneMPReg>();
    }
}

void NoneMPReg::registerMP( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2, std::shared_ptr<Map3D> map_3d )
{
    //std::cerr << "ERROR: MAP POINT REGISTRATION ALGORITHM NOT IMPLEMENTED" << std::endl;
}