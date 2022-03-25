#include <memory>
#include <opencv2/opencv.hpp>

#include "nonLinReproOpt.hpp"
#include "../../util/util.hpp"

using std::vector;
using std::shared_ptr;


std::shared_ptr<Pose> ReproOpt::calculatePoseLinear( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2 )
{
    /*
    Arguments:
        frameX:     <FrameData> to frame nr X.
    Returns:
        rel_pose:   Relative pose frame frame2 to frame 1.
    Effect:
        - Removes outlier matches.
        - Assigning the rel pose to both frames.
    Assumption:
        - K_matrix equal for both frames.
    */

    cv::Mat E_matrix, inliers;
    std::vector<cv::Point> pts1, pts2;

    compileMatchedCVPoints( frame1, frame2, pts1, pts2 );
    E_matrix = cv::findEssentialMat( pts1, pts2, frame1->getKMatrix(), cv::RANSAC, 0.999, 1.0, inliers );
    FrameData::removeOutlierMatches( inliers, frame1, frame2 );

    std::shared_ptr<Pose> rel_pose = FrameData::registerRelPose( E_matrix, frame1, frame2 );

    return rel_pose;
}

void ReproOpt::kptTriangulationLinear( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2, cv::Mat& XYZ1 )
{
    /*
    Arguments: 
        T_rel:      Relative transformation from frame 2 to frame 1 [4 x 4].
        uv1_X:      Keypoint locations from frame X in homogeneous coordinates [3 x N].
        KX:         Intrinsic camera matrix of frame X [3 x 3].
    Returns: 
        XYZ1:       3D location of triangulated keypoints matched between frame 1 and frame 2, dehomogenized [4 x N].
    */

    //-------- Preparing data --------------------
    cv::Mat_<double> uv1_1, uv1_2, K1, K2, T1, T2;
    vector<shared_ptr<KeyPoint2>> kpts1, kpts2;

    K1 = frame1->getKMatrix();
    K2 = frame2->getKMatrix();
    T1 = frame1->getGlobalPose();
    T2 = frame2->getGlobalPose();
    
    copyMatchedKptsLists( frame1, frame2, kpts1, kpts2 );         //TODO: Copying is inneficient and might lead to larger overhead, change it
    uv1_1 = FrameData::compileCVPointCoords( kpts1 );
    uv1_2 = FrameData::compileCVPointCoords( kpts2 );
    std::shared_ptr<Pose> rel_pose = frame1->getRelPose(frame2);
    cv::Mat T_rel = rel_pose->getTMatrix().rowRange(0,3).colRange(0,4);
    //--------------------------------------------

    triangulatePointsLinear( T_rel, K1, K2, uv1_1, uv1_2, XYZ1 );
    dehomogenizeMatrix( XYZ1 );
}

double ReproOpt::residual( cv::Mat& T_rel, cv::Mat& K1, cv::Mat& K2, cv::Mat& XYZ1, cv::Mat& uv1_1, cv::Mat& uv1_2 )
{
    /*
    Arguments:
        T_rel:      Relative transformation from frame 2 to frame 1 [4 x 4].
        KX:         Intrinsic camera matrix of frame X [3 x 3].
        XYZ1:       3D location of triangulated keypoints matched between frame 1 and frame 2, dehomogenized [4 x N].
        uv1_X:      Dehomogenized pixel coordinates of the keypoints from frame nr X.
    Return:
        res:        Residual = sum(|| uv1_1 - pi_1(XYZ1)||^2 + || uv1_2 - pi_2(XYZ1)||^2).
    */

    int N = XYZ1.cols;
    double res_i;
    cv::Mat uv1, uv2, uv1_err, uv2_err, xyz1, res_i_mat;

    double res = 0;
    cv::Mat I_0 = cv::Mat::eye(4,4,CV_64F);
    for ( int n = 0; n < N; ++n)
    {
        uv1 = uv1_1.col(n);
        uv2 = uv1_2.col(n);
        xyz1 = XYZ1.col(n);
        uv1_err = reprojectionError( xyz1, uv1, T_rel, K1 );
        uv2_err = reprojectionError( xyz1, uv2, I_0, K2 );
        res_i_mat = uv1_err.t() * uv1_err + uv2_err.t() * uv2_err;
        res_i = res_i_mat.at<double>(0,0);
        //std::cout << uv1_err.t() << "\n * \n" << uv1_err << "\n = \n" << uv1_err.t() * uv1_err << std::endl;
        //std::cout << res_i << std::endl;
        res += res_i;
    }

    //TODO: Increase the speed by taking in addresses to the projectKpt function
    //TODO: Look into possibility of doing faster computations by doing matrix computations instead.

    return res;
}

std::shared_ptr<Pose> ReproOpt::calculate( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2 )
{
    // Function is copying out uv1_1 and uv1_2 two separate times, this will cause some significant slowdown of the
    //  process.
    std::shared_ptr<Pose> rel_pose = this->calculatePoseLinear( frame1, frame2 );

    cv::Mat XYZ1;
    this->kptTriangulationLinear( frame1, frame2, XYZ1 );

    //-------- Preparing data --------------------
    cv::Mat uv1_1, uv1_2, K1, K2, T_rel;
    vector<shared_ptr<KeyPoint2>> kpts1, kpts2;

    K1 = frame1->getKMatrix();
    K2 = frame2->getKMatrix();

    copyMatchedKptsLists( frame1, frame2, kpts1, kpts2 );         //TODO: Copying is inneficient and might lead to larger overhead, change it
    uv1_1 = FrameData::compileCVPointCoords( kpts1 );
    uv1_2 = FrameData::compileCVPointCoords( kpts2 );
    T_rel = rel_pose->getTMatrix();
    //--------------------------------------------

    this->residual( T_rel, K1, K2, XYZ1, uv1_1, uv1_2 );

    //TODO: Up until this point we are working with relative tranformations, remember that we have to revert back into global coordinates.
    //XYZ1 = T1 * XYZ1;
    //dehomogenizeMatrix( XYZ1 );

    return rel_pose;
}