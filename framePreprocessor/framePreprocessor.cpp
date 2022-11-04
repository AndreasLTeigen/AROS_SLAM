#include <opencv2/opencv.hpp>

#include "framePreprocessor.hpp"

#include "preprocessMethods/autocorrelation.hpp"
#include "preprocessMethods/blur.hpp"


std::shared_ptr<Preprocessor> getPreprocessor( std::string preprocessor_method )
{
    if ( preprocessor_method == "autocor" )
    {
        return std::make_shared<Autocor>();
    }
    else if (preprocessor_method == "blur" )
    {
        return std::make_shared<Blur>();
    }
    else
    {
        std::cerr << "ERROR: PREPROCESSOR METHOD NOT FOUND" << std::endl;
        return std::make_shared<NoneProcessor>();
    }
}

void NoneProcessor::calculate( cv::Mat& img, std::shared_ptr<FrameData> frame )
{
    std::cerr << "ERROR: KEYPOINT EXTRACTION ALGORITHM NOT IMPLEMENTED" << std::endl;
}