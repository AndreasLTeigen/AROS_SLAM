#include <opencv2/opencv.hpp>

#include "point5OutlierRejection.hpp"
#include "../../util/util.hpp"



std::shared_ptr<Pose> P5ORPC::calculate( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2, cv::Mat& img )
{
    // Assumes K_matrix is equal for both frames.
    return this->do5pointAlgOutlierRejection( frame1, frame2, frame1->getKMatrix() );
}

void P5ORPC::analysis( cv::Mat &img_disp, std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2 )
{
    std::cerr << "ERROR: POSE CALCULATION ANALYSIS ALGORITHM NOT IMPLEMENTED" << std::endl;
}

std::shared_ptr<Pose> P5ORPC::do5pointAlgOutlierRejection(std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2, cv::Mat K_matrix)
{
    /*  Arguments:
            frame1:                 Current frame.
            frame2:                 Previous frame.
            K_matrix:               Camera calibration matrix of both cameras.
            pose_calculation_type:  Method of calculating relative pose.
        Returns:
            rel_pose:               Copy of the relative pose.

        Explanation:
            Calculates relative pose and references the new pose object in frame 1 and frame 2.
    */

    cv::Mat E_matrix, inliers;
    std::vector<cv::Point> pts1, pts2;

    compileMatchedCVPoints(frame1, frame2, pts1, pts2);

    if ( pts1.size() < 5 || pts2.size() < 5 )
    {
        std::cerr << "ERROR: Fewer then 5 matched points are available, cannot do pose calculation" << std::endl;
        return nullptr;
    }
    E_matrix = cv::findEssentialMat(pts1, pts2, K_matrix, cv::RANSAC, 0.999, 1.0, inliers);
    FrameData::removeOutlierMatches(inliers, frame1, frame2);

    std::shared_ptr<Pose> rel_pose = FrameData::registerRelPose(E_matrix, frame1, frame2);
    return rel_pose;
}