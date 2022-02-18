#ifndef CopyMPPC_h
#define CopyMPPC_h


#include "../poseCalculation.hpp"
#include "../../dataStructures/frameData.hpp"
#include "../../dataStructures/pose.hpp"


class CopyMPPC : public PoseCalculator    // 5-Point with Outlier Rejection Pose Calculator
{
    public:
        CopyMPPC(){};
        ~CopyMPPC(){};

        std::shared_ptr<Pose> calculate( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2, cv::Mat K_matrix )override;
};

#endif