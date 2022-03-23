#include <memory>
#include <opencv2/opencv.hpp>

#include "GJET.hpp"
#include "../../util/util.hpp"

using std::vector;
using std::shared_ptr;


//double calculateLagrangian(vector<shared_ptr<KeyPoint2>> matched_kpts1, cv::Mat F_matrix)

std::shared_ptr<Pose> GJET::calculate( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2, cv::Mat K_matrix )
{
    cv::Mat E_matrix, F_matrix, inliers;
    std::vector<cv::Point> pts1, pts2;

    compileMatchedCVPoints( frame1, frame2, pts1, pts2 );
    E_matrix = cv::findEssentialMat( pts1, pts2, K_matrix, cv::RANSAC, 0.999, 1.0, inliers );
    FrameData::removeOutlierMatches( inliers, frame1, frame2 );
    F_matrix = fundamentalFromEssential( E_matrix, frame1->getKMatrix(), frame2->getKMatrix() );

    vector<shared_ptr<KeyPoint2>> matched_kpts1 = frame1->getMatchedKeypoints( frame2->getFrameNr() );

    std::shared_ptr<Pose> rel_pose = FrameData::registerRelPose( E_matrix, frame1, frame2 );
    return rel_pose;
}