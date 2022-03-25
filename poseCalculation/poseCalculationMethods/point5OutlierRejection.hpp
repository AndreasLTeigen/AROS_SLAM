#ifndef P5ORPC_h
#define P5ORPC_h

#include "../poseCalculation.hpp"
#include "../../dataStructures/frameData.hpp"
#include "../../dataStructures/pose.hpp"


class P5ORPC : public PoseCalculator    // 5-Point with Outlier Rejection Pose Calculator
{
    public:
        P5ORPC(){};
        ~P5ORPC(){};

        std::shared_ptr<Pose> calculate( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2 )override;
        std::shared_ptr<Pose> do5pointAlgOutlierRejection(std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2, cv::Mat K_matrix);

};

#endif