#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

#include "opticalFlowFarneback.hpp"
#include "../../dataStructures/frameData.hpp"
#include "../../dataStructures/match.hpp"
#include "../../dataStructures/keypoint.hpp"

using cv::Mat;
using std::vector;

int OpticalFlowFarneback::matchKeypoints( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2 )
{
    cv::Mat img1, img2;

    img1 = frame1->getImg();
    img2 = frame2->getImg();

    // Calculate optical flow
    cv::Mat flow(img2.size(), CV_32FC2);
    cv::calcOpticalFlowFarneback(   img2, img1, flow, 
                                    this->pyr_scale, 
                                    this->levels, 
                                    this->winsize, 
                                    this->iterations, 
                                    this->poly_n, 
                                    this->poly_sigma, 
                                    this->flags);

    //this->visualizeFlow(img1, flow);


    // Converting flow to points.
    Mat flow_parts[2], magnitude, inliers, K_matrix, E_matrix;
    cv::split(flow, flow_parts);
    cv::sqrt((flow_parts[0].mul(flow_parts[0]) + flow_parts[1].mul(flow_parts[1])), magnitude);

    magnitude.setTo(0, magnitude < this->flow_threshold);
    
    float flow_x, flow_y;
    vector<cv::Point2d> pts1, pts2;
    for ( int row_i = 0; row_i < flow.rows; ++row_i )
    {
        for ( int col_j = 0; col_j < flow.cols; ++col_j )
        {
            if ( magnitude.at<float>(row_i, col_j) != 0 )
            {
                flow_y = flow_parts[1].at<float>(row_i, col_j);
                flow_x = flow_parts[0].at<float>(row_i, col_j);
                pts1.push_back(cv::Point2d(float(col_j) + flow_x, float(row_i) + flow_y));
                pts2.push_back(cv::Point2d(float(col_j), float(row_i)));
                //std::cout << flow_x << std::endl;
                //std::cout << float(pts2.back().x) << std::endl;
                //std::cout << pts1.back().x << std::endl;
                //std::cout << "-------------" << std::endl;
            }
        }
    }

    /*
    frame1->registerKeypoints(pts1);
    frame2->registerKeypoints(pts2);
    
    vector<std::shared_ptr<KeyPoint2>> kpts1, kpts2;
    kpts1 = frame1->getKeypoints();
    kpts2 = frame2->getKeypoints();

    std::shared_ptr<KeyPoint2> kpt1, kpt2;
    for ( int i = 0; i < pts1.size(); ++i )
    {
        if (i%1000 == 0)
        {
            kpt1 = kpts1[i];
            kpt2 = kpts2[i];
            // Registering the match.
            std::shared_ptr<Match> match = std::shared_ptr<Match>(new Match(kpt1, kpt2, 0, i));
            kpt1->addMatch(match, frame2->getFrameNr());
            kpt2->addMatch(match, frame1->getFrameNr());

            frame1->addKptToMatchList(kpt1, frame2);
            frame2->addKptToMatchList(kpt2, frame1);
        }
    }
    */

    K_matrix = frame1->getKMatrix();

    E_matrix = cv::findEssentialMat(pts1, pts2, K_matrix, cv::RANSAC, 0.999, 1.0, inliers);
    //E_matrix = cv::findEssentialMat(pts1, pts2, K_matrix, cv::RANSAC, 0.999, 1.0, 1, inliers);
    //FrameData::removeOutlierMatches(inliers, frame1, frame2);
    std::shared_ptr<Pose> rel_pose = std::make_shared<Pose>(E_matrix, frame1, frame2, pts1, pts2);
    FrameData::registerRelPose(rel_pose, frame1, frame2);
    std::cout << rel_pose->getTMatrix() << std::endl;

    return 0;
}

void OpticalFlowFarneback::visualizeFlow(cv::Mat& img, cv::Mat& flow)
{
    cv::Mat flow_parts[2];
    cv::split(flow, flow_parts);
    cv::Mat magnitude, angle, magn_norm;
    cv::cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
    cv::normalize(magnitude, magn_norm, 0.0f, 1.0f, cv::NORM_MINMAX);
    angle *= ((1.f / 360.f) * (180.f / 255.f));

    //build hsv image
    cv::Mat _hsv[3], hsv, hsv8, bgr;
    _hsv[0] = angle;
    _hsv[1] = cv::Mat::ones(angle.size(), CV_32F);
    _hsv[2] = magn_norm;
    cv::merge(_hsv, 3, hsv);
    hsv.convertTo(hsv8, CV_8U, 255.0);
    cv::cvtColor(hsv8, bgr, cv::COLOR_HSV2BGR);
    cv::imshow("frame2", bgr);
    cv::waitKey(0);
}