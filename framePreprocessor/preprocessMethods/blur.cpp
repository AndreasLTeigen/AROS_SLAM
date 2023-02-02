#include <opencv2/opencv.hpp>

#include "blur.hpp"



void Blur::calculate( cv::Mat& img )
{
    cv::Mat out;
    cv::GaussianBlur(img, out, kernel_size, 0);
    img = out;
}