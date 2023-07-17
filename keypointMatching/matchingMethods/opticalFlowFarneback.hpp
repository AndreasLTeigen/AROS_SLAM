#ifndef farneback_h
#define farneback_h

#include "../matchKeypoints.hpp"
#include "../../dataStructures/frameData.hpp"

class OpticalFlowFarneback : public Matcher
{
    private:
        double pyr_scale = 0.5;
        int levels = 3;
        int winsize = 15;
        int iterations = 3;
        int poly_n = 5;
        double poly_sigma = 1.2;
        int flags = 0;

        double flow_threshold = 1;

    public:
        OpticalFlowFarneback(){};
        ~OpticalFlowFarneback(){};

        int matchKeypoints( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2 )override;
        void visualizeFlow( cv::Mat& img, cv::Mat& flow );
};

#endif