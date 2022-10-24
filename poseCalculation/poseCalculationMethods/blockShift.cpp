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

    /*
    kpts = frame2->getKeypoints();
    for ( int i = 0; i < kpts.size(); ++i )
    {
        kpt = kpts[i];
        //Getting the previous keypoints.
        pts2.push_back( cv::Point(kpt->getCoordX(), kpt->getCoordY()) );

        // Getting the shifted keypoints.
        shift = kpt->getDescriptor("shift");
        if ( shift.empty() )
        {
            std::cerr << "ERROR: SHIFT VALUE IS EMPTY " << std::endl;
        }
        pts1.push_back( cv::Point(shift.at<double>(0,0), shift.at<double>(1,0)) );
    }
    */

    kpts1 = frame2->getKeypoints();
    kpts2 = frame2->getKeypoints();

    for ( int i = 0; i < kpts1.size(); ++i )
    {
        kpt1 = kpts1[i];
        kpt2 = kpts2[i];

        center = kpt2->getDescriptor("center");
        kpt2->setCoordx(center.at<double>(0,0));
        kpt2->setCoordy(center.at<double>(1,0));
    }



    compileMatchedCVPoints(frame1, frame2, pts1, pts2);

    K_matrix = frame2->getKMatrix();

    E_matrix = cv::findEssentialMat(pts1, pts2, K_matrix, cv::RANSAC, 0.999, 1.0, inliers);
    FrameData::removeOutlierMatches(inliers, frame1, frame2);

    std::shared_ptr<Pose> rel_pose = FrameData::registerRelPose(E_matrix, frame1, frame2);

    return rel_pose;
}

void BlockShift::analysis( cv::Mat &img_disp, std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2 )
{
    std::cerr << "ERROR: POSE CALCULATION ANALYSIS ALGORITHM NOT IMPLEMENTED" << std::endl;
}