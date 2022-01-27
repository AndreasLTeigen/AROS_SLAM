#include <opencv2/opencv.hpp>

#include "poseCalculation.hpp"
#include "poseCalculationMethods/point5OutlierRejection.hpp"


PoseCalculator getRelativePoseCalculationMethod( std::string pose_calculation_method )
{
    /* Retrieve the right pose calculator class from the corresponding config string */

    if ( pose_calculation_method == "5-point + outlier removal" )
    {
        // Brute force KNN matching of keypoints with lowes ratio test
        return PoseCalculator::P5OR;
    }
    else
    {
        std::cout << "ERROR: RELATIVE POSE CALCULATION METHOD NOT FOUND" << std::endl;
        return PoseCalculator::NONE;
    }
}

std::shared_ptr<Pose> calculateRelativePose(std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2, cv::Mat K_matrix, PoseCalculator pose_calculation_type)
{
    /*  Arguments:
            frame1:                 Current frame
            frame2:                 Previous frame
            K_matrix:               Camera calibration matrix of both cameras
            pose_calculation_type:  Method of calculating relative pose
        Returns:
            rel_pose:               Copy of the relative pose

        Explanation:
            Calculates relative pose and references the new pose object in frame 1 and frame 2 
    */

    std::shared_ptr<Pose> rel_pose;

    switch(pose_calculation_type)
    {
        case PoseCalculator::P5OR:
        {
            rel_pose = do5pointAlgOutlierRejection( frame1, frame2, K_matrix );
        } break;

        default:
        {
            std::cout << "ERROR: RELATIVE POSE ALGORITHM NOT IMPLEMENTED" << std::endl;
        }
    }
    return rel_pose;
}