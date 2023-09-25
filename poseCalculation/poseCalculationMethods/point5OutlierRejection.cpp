#include <opencv2/opencv.hpp>

#include "point5OutlierRejection.hpp"
#include "../../util/util.hpp"



int P5ORPC::calculate(std::shared_ptr<FrameData> frame1, 
                                        std::shared_ptr<FrameData> frame2, 
                                        cv::Mat& img )
{
    // Assumes K_matrix is equal for both frames.
    return this->do5pointAlgOutlierRejection(   frame1, frame2, 
                                                frame1->getKMatrix() );
}

int P5ORPC::do5pointAlgOutlierRejection(std::shared_ptr<FrameData> frame1, 
                                        std::shared_ptr<FrameData> frame2, 
                                        cv::Mat K_matrix)
{
    /*  
    Calculates relative pose and references the new pose object in frame 1 and 
    frame 2.

    Arguments:
        frame1:                 Current frame.
        frame2:                 Previous frame.
        K_matrix:               Camera calibration matrix of both cameras.
        pose_calculation_type:  Method of calculating relative pose.
    Returns:
        rel_pose:               Copy of the relative pose.
    */

    cv::Mat E_matrix, inliers;
    std::vector<cv::Point> pts1, pts2;

    compileMatchedCVPoints(frame1, frame2, pts1, pts2);

    // Drops frame if less than 5 matched points are found.
    if ( pts1.size() <= 5 || pts2.size() <= 5 )
    {
        std::cerr << "WARNING: Fewer then 5 matched points are available, cannot do pose calculation.\nRecovering..." << std::endl;
        return 2;
    }

    // Drops frame if keypoints have moved less than some distance.
    if (this->do_stationary_frame_skip)
    {
        if (this->isStationaryFrame(pts1, pts2))
        {
            std::cerr << "WARNING: Too little paralax present, dropping frame\nRecovering..." << std::endl;
            return 2;
        }
    }

    // E_matrix = cv::findEssentialMat(pts1, pts2, K_matrix, cv::RANSAC, 0.999, 
    //                                 1.0, inliers);

    E_matrix = cv::findEssentialMat(pts2, pts1, K_matrix, cv::USAC_DEFAULT, 
                                    0.999, 1.0, inliers);

    // If no motion hypothesis with 5 or more points can be established,
    // E_matrix is returned as empty.
    if (E_matrix.empty())
    {
        return 2;
    }


    //this->num_outliers = int(frame1_points.cols - cv::sum(inliers)[0]/255.0);
    this->num_inliers = cv::sum(inliers)[0];
    this->num_outliers = pts1.size() - this->num_inliers;

    if (this->remove_outliers)
    {
        FrameData::removeOutlierMatches(inliers, frame1, frame2);
    }

    std::shared_ptr<Pose> rel_pose = 
                FrameData::registerRelPose(E_matrix, frame1, frame2);
    return 0;
}