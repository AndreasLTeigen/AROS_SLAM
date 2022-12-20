#include <vector>
#include <opencv2/opencv.hpp>

#include "bruteForceMatching.hpp"
#include "../../dataStructures/frameData.hpp"

using cv::Mat;
using cv::DMatch;
using std::vector;
using std::shared_ptr;


void BFMatcher::matchKeypoints( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2 )
{
    //--Performing brute force matching without cross check and normalized hamming distance

    if (frame1->getNumKeypoints() == 0 )
    {
        std::cerr << "ERROR: No keypoints were found" << std::endl;
    }
    
    vector<vector<DMatch>> matches, matches_temp;
    Mat queryDesc = frame1->compileCVDescriptors();
    Mat trainDesc = frame2->compileCVDescriptors();

    this->matcher.knnMatch(queryDesc, trainDesc, matches_temp, 2);

    //--Lowes ratio test
    for( int i = 0; i < matches_temp.size(); i++)
    {
        if ( matches_temp[i][0].distance < 0.8*matches_temp[i][1].distance)
        {
            vector<DMatch> temp;
            temp.push_back(matches_temp[i][0]);
            matches.push_back(temp);
        }
    }
    FrameData::registerMatches(frame1, frame2, matches);

    if (this->is_logging)
    {
        this->num_match_curr = frame1->getMatchedKeypoints(frame2->getFrameNr()).size();
    }
}