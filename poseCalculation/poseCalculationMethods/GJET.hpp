#ifndef GJET_h
#define GJET_h

#include "../poseCalculation.hpp"
#include "../../dataStructures/frameData.hpp"
#include "../../dataStructures/pose.hpp"


class GJET : public PoseCalculator    // 5-Point with Outlier Rejection Pose Calculator
{
    public:
        GJET(){};
        ~GJET(){};

        std::shared_ptr<Pose> calculate( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2, cv::Mat K_matrix )override;
};

#endif