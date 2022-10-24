#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

#include "matchKeypoints.hpp"
#include "matchingMethods/bruteForceMatching.hpp"
#include "matchingMethods/phaseCorrelation.hpp"

using std::string;
using std::shared_ptr;

std::shared_ptr<Matcher> getMatcher( string matching_method )
{
    if ( matching_method == "bf_mono" )
    {
        return std::make_shared<BFMatcher>();
    }
    else if ( matching_method == "phaseCorr")
    {
        return std::make_shared<PhaseCorrelation>();
    }
    else
    {
        std::cerr << "ERROR: EXTRACTION METHOD NOT FOUND" << std::endl;
        return std::make_shared<NoneMatcher>();
    }
}




void NoneMatcher::matchKeypoints( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2 )
{
    std::cerr << "ERROR: KEYPOINT MATCHING ALGORITHM NOT IMPLEMENTED" << std::endl;
}