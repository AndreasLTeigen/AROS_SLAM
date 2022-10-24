#ifndef phaseCorrelation_h
#define phaseCorrelation_h

#include <opencv2/opencv.hpp>

#include "../matchKeypoints.hpp"
#include "../../dataStructures/frameData.hpp"


class PhaseCorrelation : public Matcher
{
    public:
        PhaseCorrelation(){};
        ~PhaseCorrelation(){};

        void matchKeypoints( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2 )override;
};


#endif