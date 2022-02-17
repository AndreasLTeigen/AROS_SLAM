#ifndef bruteForceMatching_h
#define bruteForceMatching_h

#include <opencv2/opencv.hpp>

#include "../matchKeypoints.hpp"
#include "../../dataStructures/frameData.hpp"


class BFMatcher : public Matcher
{
    private:
        cv::BFMatcher matcher = cv::BFMatcher(cv::NORM_HAMMING, false);

    public:
        BFMatcher(){};
        ~BFMatcher(){};

        void matchKeypoints( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2 )override;
};

#endif