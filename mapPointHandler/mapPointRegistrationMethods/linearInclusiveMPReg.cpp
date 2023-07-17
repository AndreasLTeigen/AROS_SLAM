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


int LinIncMPReg::registerMP( std::shared_ptr<FrameData> frame1, 
                              std::shared_ptr<FrameData> frame2, 
                              std::shared_ptr<Map3D> map_3d )
{
    /*
    Linearly triangulates every matched keypoint between frame 1 and 2 and 
            adds them as map point to the map.
            
    TODO: Make functionality to update existing map points instead of always
            adding new ones.
    
    Arguments:
        frameX:     Frames whose matched points should be used to update 
                    <map_3d>
        map_3d:     3D global map

    Returns:
        err:        If parts of the code did not execute as predicted, 
                    return 1, otherwise return 0.
 
    TODO: Clean up linearInclusiveMPReg( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2, std::shared_ptr<Map3D> map_3d )
    */

    cv::Mat_<double> uv1, uv2, K1, K2, T1, T2, X, uncertainty_3D;
    vector<shared_ptr<KeyPoint2>> kpts1, kpts2;
    K1 = frame1->getKMatrix();
    K2 = frame2->getKMatrix();
    T1 = frame1->getGlobalPose();
    T2 = frame2->getGlobalPose();

    copyMatchedKptsLists( frame1, frame2, kpts1, kpts2 );         //TODO: Copying is inneficient and might lead to larger overhead, change it
    uv1 = FrameData::compilePointCoords( kpts1 );
    uv2 = FrameData::compilePointCoords( kpts2 );

    std::shared_ptr<Pose> rel_pose = frame1->getRelPose(frame2);
    cv::Mat rel_T = rel_pose->getTMatrix().rowRange(0,3).colRange(0,4);

    triangulatePointsLinear( rel_T, K1, K2, uv1, uv2, X ); //X_c1

    dehomogenizeMatrix( X );

    int err = this->removeInvalidPoints( X );
    if (err)
    {
        std::cerr << "Warning(frame " << frame1->getImgId() << "): All triangulated points behind camera.\n" 
            << "Skipping map update.." << std::endl;
        return 1;
    }

    X = T1 * X; // X_w

    uncertainty_3D = Map3D::calculate3DUncertainty(X, uv1, uv2, K1, K2, T1, T2); //TODO: Implement this function
    
    map_3d->batchUpdateMap( kpts1, kpts2, T1, T2, X, uncertainty_3D);
}


int LinIncMPReg::removeInvalidPoints( cv::Mat& X_c )
{
    /* Removes points that are invalid (are located behind camera) */

    cv::Mat invalid_mask;
    this->isPointBehindCamera( X_c, invalid_mask );
    removeColumns( X_c, invalid_mask );
    if (X_c.cols == 0)
    {
        return 1;
    }
    return 0;
}

void LinIncMPReg::isPointBehindCamera( cv::Mat& X_c, cv::Mat& ret )
{
    /*
    Returns true if point is behind camera in a boolean list aligned with X_c

    Arguments:
        X_c:    Homogeneous 3D points in camera coordinates [4 x N]
    Returns:
        ret:    Return mask: True if point is behind camera, false otherwise.
    */
    int num_rows = X_c.rows;

    ret = X_c.row(num_rows-2) < 0;
}