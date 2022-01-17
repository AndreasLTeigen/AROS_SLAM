#ifndef triangulateMany_h
#define triangulateMany_h

#include <vector>
#include <opencv2/opencv.hpp>

cv::Mat linearTriangulateMany(cv::Mat& uv1, cv::Mat& uv2, cv::Mat& K1, cv::Mat& K2, cv::Mat& T1, cv::Mat& T2);

int opencvTriangulationTest(cv::InputArray _points1, cv::InputArray _points2, cv::InputArray _cameraMatrix, cv::InputArray _P1, cv::InputArray _P2, cv::OutputArray triangulatedPoints);

void triangulateTest(cv::Mat K1, cv::Mat K2);

#endif