#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>

#include "framePreprocessor.hpp"

#include "preprocessMethods/autocorrelation.hpp"
#include "preprocessMethods/noise.hpp"
#include "preprocessMethods/blur.hpp"
#include "preprocessMethods/fft.hpp"
#include "preprocessMethods/homomorphicFilter.hpp"


std::shared_ptr<Preprocessor> getPreprocessor( YAML::Node config )
{
    std::string preprocessor_method = config["Method.preprocessor"].as<std::string>();
    const YAML::Node pre_pro_config = config["Pre-processing"];
    
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
        return std::make_shared<Blur>(pre_pro_config["Blur"]);
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
        return std::make_shared<NoneProcessor>();
    }
}

NoneProcessor::NoneProcessor()
{
    std::cout << std::left;
    std::cout << std::setw(20) << "Pre-processor:" << "None" << std::endl; 
}

void NoneProcessor::calculate( cv::Mat& img )
{
    //std::cerr << "ERROR: KEYPOINT EXTRACTION ALGORITHM NOT IMPLEMENTED" << std::endl;
}