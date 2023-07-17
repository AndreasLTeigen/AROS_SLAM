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

void MapPointRegistrator::analysis( std::shared_ptr<FrameData> frame1, 
                                    std::shared_ptr<FrameData> frame2, 
                                    std::shared_ptr<Map3D> map_3d )
{
    /* 
    Calculates the reprojection error of all keypoints of <frame1> measured 
    against the corresponding <MapPoint>s of <map_3d>.
    
    Arguments:
        frameX:     Frames whose matched points should be used to update 
                    <map_3d>
        map_3d:     3D global map
    */

    cv::Mat_<double> uv1, K1, T1, X, point_loc_3d, XYZ1;
    vector<shared_ptr<KeyPoint2>> kpts1;
    shared_ptr<MapPoint> map_point;
    K1 = frame1->getKMatrix();
    T1 = frame1->getGlobalPose();

    int i = 0;
    kpts1 = frame1->getKeypoints();
    XYZ1 = cv::Mat::zeros(4, kpts1.size(), CV_64F);

    for (shared_ptr<KeyPoint2> kpt1 : kpts1)
    {
        map_point = kpt1->getMapPoint();
        if (map_point != nullptr)
        {
            point_loc_3d = map_point->getCoordXYZ1();
            point_loc_3d.copyTo(XYZ1.col(i));
            i += 1;
        }
    }
    //std::cout << XYZ1 << std::endl;
}

int NoneMPReg::registerMP(  std::shared_ptr<FrameData> frame1, 
                            std::shared_ptr<FrameData> frame2, 
                            std::shared_ptr<Map3D> map_3d )
{
    //std::cerr << "ERROR: MAP POINT REGISTRATION ALGORITHM NOT IMPLEMENTED" << std::endl;
    return 0;
}