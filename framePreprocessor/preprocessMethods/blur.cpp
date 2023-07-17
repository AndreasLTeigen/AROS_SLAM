#include <iostream>
#include <opencv2/opencv.hpp>

#include "blur.hpp"


Blur::Blur(const YAML::Node config)
{
    std::cout << std::left;
    std::cout << std::setw(20) << "Pre-processor:" << " Blur" << std::endl; 
    int k_size = config["kernel_size"].as<int>();
    if (k_size % 2 == 0 || k_size == 0)
    {
        std::cerr << "ERROR: Blur kernel size has to be an odd integer bigger than zero" << std::endl;
    }
    this->kernel_size = cv::Size(k_size, k_size);
}

void Blur::calculate( cv::Mat& img )
{
    cv::Mat out;
    cv::GaussianBlur(img, out, kernel_size, 0);
    img = out;
}