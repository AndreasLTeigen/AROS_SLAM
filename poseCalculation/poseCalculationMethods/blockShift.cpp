#include <opencv2/opencv.hpp>

#include "blockShift.hpp"
#include "../../util/util.hpp"


using std::vector;
using std::shared_ptr;

std::shared_ptr<Pose> BlockShift::calculate( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2, cv::Mat& img )
{
    // Assumes K_matrix is equal for both frames.
    cv::Mat E_matrix, K_matrix, inliers, center;
    vector<cv::Point> pts1, pts2;
    shared_ptr<KeyPoint2> kpt1, kpt2;
    vector<shared_ptr<KeyPoint2>> kpts1, kpts2;

    compileMatchedCVPoints(frame1, frame2, pts1, pts2);

    if ( pts1.size() <= 5 )
    {
        resetKptMatches(frame1, frame2);
        return nullptr;
    }

    K_matrix = frame2->getKMatrix();

    E_matrix = cv::findEssentialMat(pts1, pts2, K_matrix, cv::RANSAC, 0.999, 1.0, inliers);
    FrameData::removeOutlierMatches(inliers, frame1, frame2);
    std::shared_ptr<Pose> rel_pose = FrameData::registerRelPose(E_matrix, frame1, frame2);

    return rel_pose;
}

void BlockShift::resetKptMatches( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2 )
{
    // Removes all matches between frame1 and frame2.
    FrameData::removeMatchesWithLowConfidence(1, frame1, frame2);
}

void BlockShift::analysis( cv::Mat &img_disp, std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2 )
{
    std::cerr << "ERROR: POSE CALCULATION ANALYSIS ALGORITHM NOT IMPLEMENTED" << std::endl;
}