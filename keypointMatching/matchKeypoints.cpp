#include <vector>
#include <memory>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>

#include "matchKeypoints.hpp"
#include "matchingMethods/bruteForceMatching.hpp"
#include "matchingMethods/phaseCorrelation.hpp"
#include "matchingMethods/opticalFlowFarneback.hpp"

#include "../util/util.hpp"

using std::string;
using std::shared_ptr;

std::shared_ptr<Matcher> getMatcher( YAML::Node config )
{
    std::string matching_method = config["Method.matcher"].as<std::string>();
    const YAML::Node matcher_config = config["Matcher"];

    if ( matching_method == "BFMatcher" )
    {
        return std::make_shared<BFMatcher>(matcher_config["BFMatcher"]);
    }
    else if ( matching_method == "phaseCorr" )
    {
        return std::make_shared<PhaseCorrelation>();
    }
    else if ( matching_method == "KLT" )
    {
        return std::make_shared<KLTTracker>();
    }
    else if ( matching_method == "OF_F")
    {
        return std::make_shared<OpticalFlowFarneback>();
    }
    else
    {
        return std::make_shared<NoneMatcher>();
    }
}

Matcher::Matcher()
{
    if (this->analysis_match_count)
    {
        clearFile(f_match_count);
    }
}

void Matcher::analysis(   std::shared_ptr<FrameData> frame1, 
                            std::shared_ptr<FrameData> frame2 )
{
    if (this->analysis_match_count)
    {
        writeInt2File(this->f_match_count, this->num_matches);
    }
}

int Matcher::getCurrMatchNum()
{
    return this->num_matches;
}

NoneMatcher::NoneMatcher()
{
    std::cout << std::left;
    std::cout << std::setw(20) << "Matcher:" << "None" << std::endl; 
    std::cerr << "ERROR: Matcher required." << std::endl;
}

int NoneMatcher::matchKeypoints(std::shared_ptr<FrameData> frame1, 
                                std::shared_ptr<FrameData> frame2 )
{
    //std::cerr << "ERROR: KEYPOINT MATCHING ALGORITHM NOT IMPLEMENTED" << std::endl;
    return 0;
}