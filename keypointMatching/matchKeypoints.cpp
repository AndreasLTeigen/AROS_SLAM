#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

#include "matchKeypoints.hpp"
#include "matchingMethods/bruteForceMatching.hpp"

using std::vector;
using std::shared_ptr;
using cv::Mat;
using cv::DMatch;
using cv::KeyPoint;

Matcher getMatchingMethod( std::string matching_method )
{
    /* Retrieve the right keypoint matching class from the corresponding config string */

    if ( matching_method == "bf_mono" )
    {
        // Brute force KNN matching of keypoints with lowes ratio test
        return Matcher::brute_force_mono;
    }
    else
    {
        std::cout << "ERROR: MATCHING METHOD NOT FOUND" << std::endl;
        return Matcher::NONE;
    }
}

void matchKeypoints( shared_ptr<FrameData> frame1, shared_ptr<FrameData> frame2, Matcher matcher_type )
{
    switch(matcher_type)
    {
        case Matcher::brute_force_mono:
        {
            doBruteForceMatchingMono( frame1, frame2 );
        } break;

        default:
        {
            std::cout << "ERROR: KEYPOINT MATCHING ALGORITHM NOT IMPLEMENTED" << std::endl;
        }
    }
}