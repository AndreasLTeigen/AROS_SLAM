#ifndef bruteForceMatching_h
#define bruteForceMatching_h

#include <opencv2/opencv.hpp>

#include "../matchKeypoints.hpp"
#include "../../dataStructures/frameData.hpp"


class BFMatcher : public Matcher
{
    private:
        bool do_lowes_ratio_test;
        int retain_N_best_matches;
        cv::BFMatcher matcher = cv::BFMatcher(cv::NORM_HAMMING, false);

    public:
        BFMatcher(const YAML::Node config);
        ~BFMatcher(){};

        int matchKeypoints( std::shared_ptr<FrameData> frame1, 
                            std::shared_ptr<FrameData> frame2 )override;
        static void matchPruning(   std::shared_ptr<FrameData> frame1, 
                                    std::shared_ptr<FrameData> frame2,
                                    int N_remaining);
};

#endif