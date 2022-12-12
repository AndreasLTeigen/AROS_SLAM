#include <opencv2/opencv.hpp>

#include "framePreprocessor.hpp"

#include "preprocessMethods/autocorrelation.hpp"
#include "preprocessMethods/noise.hpp"
#include "preprocessMethods/blur.hpp"
#include "preprocessMethods/fft.hpp"
#include "preprocessMethods/homomorphicFilter.hpp"


std::shared_ptr<Preprocessor> getPreprocessor( std::string preprocessor_method )
{
    if ( preprocessor_method == "autocor" )
    {
        return std::make_shared<Autocor>();
    }
    else if ( preprocessor_method == "noise" )
    {
        return std::make_shared<Noise>();
    }
    else if ( preprocessor_method == "blur" )
    {
        return std::make_shared<Blur>();
    }
    else if ( preprocessor_method == "fft" )
    {
        return std::make_shared<FFT>();
    }
    else if (preprocessor_method == "hf")
    {
        return std::make_shared<HomomorphicFiltering>();
    }
    else
    {
        std::cerr << "Warning: Preprocessor method not found." << std::endl;
        return std::make_shared<NoneProcessor>();
    }
}

void NoneProcessor::calculate( cv::Mat& img, std::shared_ptr<FrameData> frame )
{
    //std::cerr << "ERROR: KEYPOINT EXTRACTION ALGORITHM NOT IMPLEMENTED" << std::endl;
}