#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "linearInclusiveMPReg.hpp"
#include "../../util/util.hpp"
#include "../../dataStructures/frameData.hpp"
#include "../../dataStructures/map3D.hpp"

using std::vector;
using std::shared_ptr;

void linearInclusiveMPReg( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2, std::shared_ptr<Map3D> map_3d )
{
    /*
    Arguments:
        frameX:     Frames whose matched points should be used to update <map_3d>
        map_3d:     3D global map
    Effect:
        map_3d->map_points: Adds new matches by creating new <mapPoint>s or updating existing <mapPoint>s
    Overview: Linearly triangulates every matched keypoint between frame 1 and 2.
    TODO: Clean up linearInclusiveMPReg( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2, std::shared_ptr<Map3D> map_3d )
    */

    cv::Mat_<double> uv1, uv2, K1, K2, T1, T2, coordinates_3D, uncertainty_3D;
    vector<shared_ptr<KeyPoint2>> kpts1, kpts2;
    K1 = frame1->getKMatrix();
    K2 = frame2->getKMatrix();
    T1 = frame1->getGlobalPose();
    T2 = frame2->getGlobalPose();

    copyMatchedKptsLists( frame1, frame2, kpts1, kpts2 );         //TODO: Copying is inneficient and might lead to larger overhead, change it
    uv1 = FrameData::compileCVPointCoords( kpts1 );
    uv2 = FrameData::compileCVPointCoords( kpts2 );

    uv1 = uv1.rowRange(0,2);
    uv2 = uv2.rowRange(0,2);

    double fx1 = K1.at<double>(0,0);
    double fy1 = K1.at<double>(1,1);
    double cx1 = K1.at<double>(0,2);
    double cy1 = K1.at<double>(1,2);

    double fx2 = K2.at<double>(0,0);
    double fy2 = K2.at<double>(1,1);
    double cx2 = K2.at<double>(0,2);
    double cy2 = K2.at<double>(1,2);

    uv1.row(0) = (uv1.row(0) - cx1) / fx1;
    uv2.row(0) = (uv2.row(0) - cx2) / fx2;
    uv1.row(1) = (uv1.row(1) - cy1) / fy1;
    uv2.row(1) = (uv2.row(1) - cy2) / fy2;


    std::shared_ptr<Pose> rel_pose = frame1->getRelPose(frame2);
    cv::Mat rel_T = rel_pose->getTMatrix().rowRange(0,3).colRange(0,4);
    cv::triangulatePoints(cv::Mat::eye(3,4, CV_64F), rel_T, uv1, uv2, coordinates_3D);
    coordinates_3D = T1 * coordinates_3D;
    dehomogenizeMatrix( coordinates_3D );

    uncertainty_3D = Map3D::calculate3DUncertainty(coordinates_3D, uv1, uv2, K1, K2, T1, T2); //TODO: Implement this function
    
    map_3d->batchUpdateMap( kpts1, kpts2, T1, T2, coordinates_3D, uncertainty_3D);
}