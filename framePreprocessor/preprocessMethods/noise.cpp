#include <opencv2/opencv.hpp>

#include "noise.hpp"



void Noise::calculate( cv::Mat& img )
{
    //cv::Mat noise(img.size(), img.type());
    cv::Mat out;
    img.convertTo(out, CV_32F);
    cv::Mat noise(img.size(), CV_32F);
    cv::randn(noise, this->mean, this->std);
    out += noise;
    cv::threshold(out, out, 255, 255, cv::THRESH_TRUNC);
    cv::threshold(out, out, 0, 255, cv::THRESH_TOZERO);
    out.convertTo(out, CV_8UC1);
    //cv::imshow("test", out);
    //cv::waitKey(0);
    img = out;
}