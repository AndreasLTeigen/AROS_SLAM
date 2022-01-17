#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "mapPointRegistration.hpp"
#include "mapPointRegistrationMethods/linearInclusiveMPReg.hpp"

using cv::KeyPoint;
using std::vector;
using std::shared_ptr;

PointReg3D get3DPointRegistrationMethod(std::string point_reg_3D_method)
{
    if ( point_reg_3D_method == "ALL" )
    {
        // All points that are matched are registered as 3D points
        return PointReg3D::ALL;
    }
    else
    {
        std::cout << "ERROR: 3D POINT REGISTRATION METHOD NOT FOUND" << std::endl;
        return PointReg3D::NONE;
    }
}


void register3DPoints( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2, std::shared_ptr<Map3D> map_3d, PointReg3D point_reg_3D)
{
    switch(point_reg_3D)
    {
        case PointReg3D::ALL:
            linearInclusiveMPReg(frame1, frame2, map_3d);
    }
}