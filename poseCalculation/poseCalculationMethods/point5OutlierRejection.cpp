#include <opencv2/opencv.hpp>

#include "point5OutlierRejection.hpp"
//#include "../../dataStructures/frameData.hpp"
//#include "../../dataStructures/pose.hpp"



std::shared_ptr<Pose> do5pointAlgOutlierRejection(std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2, cv::Mat K_matrix)
{
    cv::Mat E_matrix, inliers;
    std::vector<cv::Point> pts1, pts2;

    compileMatchedCVPoints(frame1, frame2, pts1, pts2);
    E_matrix = cv::findEssentialMat(pts1, pts2, K_matrix, cv::RANSAC, 0.999, 1.0, inliers);
    FrameData::removeOutlierMatches(inliers, frame1, frame2);

    std::shared_ptr<Pose> rel_pose = FrameData::registerRelPose(E_matrix, frame1, frame2);
    return rel_pose;
}