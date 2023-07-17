#include <opencv2/opencv.hpp>
#include <cmath>

#include "../../util/util.hpp"
#include "copyMotionPrior.hpp"


int CopyMPPC::calculate(std::shared_ptr<FrameData> frame1, 
                        std::shared_ptr<FrameData> frame2, 
                        cv::Mat& img )
{
    /* 
    Returns the already calculated motion estimation prior. 
    
    Arguments:
        frameX:     Frames between the relative pose is calculated.
        img:        Image from frame1 for reference.
    
    Returns:
        rel_pose:   Relative pose between the frames.
    */

    /* Optional: Removes all matches that does not adhere to motion model */
    if ( this->remove_outliers )
    {
        cv::Mat E_matrix, F_matrix, K1, K2, inliers;
        std::shared_ptr<Pose> rel_pose;

        rel_pose = frame1->getRelPose( frame2 );
        E_matrix = rel_pose->getEMatrix();
        F_matrix = fundamentalFromEssential(E_matrix, 
                                        frame1->getKMatrix(), 
                                        frame2->getKMatrix() ).t();
        K1 = frame1->getKMatrix();
        K2 = frame2->getKMatrix();


        cv::Mat frame1_points, frame2_points;
        compileMatchedPointCoords(  frame1, frame2, 
                                    frame1_points, frame2_points);

        inliers = cv::Mat::zeros(1,frame1_points.cols, CV_8UC1);

        getInlierMask(  F_matrix, K1, K2,
                        frame1_points, frame2_points, 
                        inliers, this->inlier_threshold);

        this->num_outliers = int(frame1_points.cols - cv::sum(inliers)[0]/255.0);
        this->num_inliers = cv::sum(inliers)[0]/255.0;

        FrameData::removeOutlierMatches(inliers, frame1, frame2);
        
        if ( frame1->getMatchedKeypoints(frame2->getFrameNr()).size() == 0 )
        {
            std::cout << "Warning: No remaining keypoints in img" << frame2->getImgId() << " and img" << frame1->getImgId() << " after outlier removal. \nRecovering...\n";
            return 2;
        }
    }

    return 0;
}