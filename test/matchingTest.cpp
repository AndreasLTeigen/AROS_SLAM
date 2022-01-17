#include <vector>
#include <chrono>
#include <memory>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "matchingTest.hpp"

using cv::Mat;
using std::string;
using std::vector;
using std::string;
using std::shared_ptr;

void matchingTest()
{
    cv::Mat F_matrix, inliers;
    string img1_str, img2_str;
    vector<cv::KeyPoint> kpts1, kpts2;
    vector<cv::Point> pts1, pts2, pts1_i, pts2_i;
    Mat descrs1, descrs2;
    vector<vector<cv::DMatch>> matches;

    img1_str = "/mnt/c/Users/and_t/Documents/AROS/Database/kitti_grayscale/dataset/sequences/00/image_0/000061.png";
    img2_str = "/mnt/c/Users/and_t/Documents/AROS/Database/kitti_grayscale/dataset/sequences/00/image_0/000062.png";

    cv::Mat img1 =  cv::imread(img1_str);
    cv::Mat img2 =  cv::imread(img2_str);

    doORBDetectAndCompute( img1, kpts1, descrs1);
    doORBDetectAndCompute( img2, kpts2, descrs2);

    doBruteForceMatchingMono(descrs2, descrs1, matches);


    std::cout << "Matches.size(): " << matches.size() << std::endl;
    
    for (int i = 0; i < matches.size(); i++)
    {
        pts1.push_back(kpts1[matches[i][0].trainIdx].pt);
        pts2.push_back(kpts2[matches[i][0].queryIdx].pt);
    }
    std::cout << "pts1.size(): " << pts1.size() << std::endl;
    std::cout << "pts2.size(): " << pts2.size() << std::endl;

    F_matrix = cv::findFundamentalMat(pts2, pts1, inliers, cv::FM_RANSAC);
    
    for (int i = 0; i < pts1.size(); i++)
    {
        if(inliers.at<uchar>(i))
        {
            //std::cout << pts1[i] << " | " << pts2[i] << std::endl;
            pts1_i.push_back(pts1[i]);
            pts2_i.push_back(pts2[i]);
        }
    }
    std::cout << "pts1_i.size(): " << pts1_i.size() << std::endl;
    std::cout << "pts2_i.size(): " << pts2_i.size() << std::endl;

    Mat img_keypoints, img_matches;
    //drawKeypoints( img1, kpts1, img_keypoints );

    //cv::drawMatches(img1, kpts1, img2, kpts2, matches, img_matches);

    drawMatchTrails(img2, pts1_i, pts2_i);

    imshow("Display window", img2);
    int k = cv::waitKey(0); // Wait for a keystroke in the window

}

void doORBDetectAndCompute(cv::Mat& frame, vector<cv::KeyPoint>& kpts,  Mat& desc)
{
    cv::Ptr<cv::ORB> detector = cv::ORB::create();
    detector->detect( frame, kpts );

    cv::Ptr<cv::ORB> descriptor = cv::ORB::create();
    descriptor->compute( frame, kpts, desc);
}

void doBruteForceMatchingMono( Mat& queryDesc, Mat& trainDesc, vector<vector<cv::DMatch>>& matches )
//--Performing brute force matching without cross check and normalized hamming distance
{
    vector<vector<cv::DMatch>> matches_temp;

    cv::BFMatcher matcher(cv::NORM_HAMMING, false);
    matcher.knnMatch(queryDesc, trainDesc, matches_temp, 2);

    //--Lowes ratio test
    for( int i = 0; i < matches_temp.size(); i++)
    {
        if ( matches_temp[i][0].distance < 0.8*matches_temp[i][1].distance)
        {
            vector<cv::DMatch> temp;
            std::cout << matches_temp[i][0].trainIdx << " | " << matches_temp[i][0].queryIdx << std::endl;
            temp.push_back(matches_temp[i][0]);
            matches.push_back(temp);
        }
    }
}

void drawMatchTrails(cv::Mat &img, vector<cv::Point> pts1, vector<cv::Point> pts2)
{
    cv::Scalar color_blue = cv::Scalar( 255, 0, 0, 128 );

    if (pts1.size() != pts2.size())
    {
        std::cout << "Size Error1" << std::endl;
    }

    for (int i = 0; i < pts1.size(); i++)
    {
        //std::cout << pts1[i] << " | " << pts2[i] << std::endl;
        cv::line(img, pts1[i], pts2[i], color_blue, 2);
    }
}