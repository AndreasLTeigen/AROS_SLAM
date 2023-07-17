#ifndef phaseCorrelation_h
#define phaseCorrelation_h

#include <opencv2/opencv.hpp>

#include "../matchKeypoints.hpp"
#include "../../dataStructures/frameData.hpp"


class PhaseCorrelation : public Matcher
{
    private:
        double shift_threshold = 0;
    public:
        PhaseCorrelation(){};
        ~PhaseCorrelation(){};

        int matchKeypoints( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2 )override;
};

class KLTTracker : public Matcher
{
    private:
        cv::Size winSize = cv::Size(21, 21);
        int maxLevel = 3;
        cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01);
        int flags = 0;
        double 	minEigThreshold = 1e-4;
        double shift_threshold = 0;

    public:
        KLTTracker(){};
        ~KLTTracker(){};

        int matchKeypoints( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2 )override;
};


#endif