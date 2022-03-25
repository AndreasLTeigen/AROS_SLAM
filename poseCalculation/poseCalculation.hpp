#ifndef poseCalculation_h
#define poseCalculation_h

#include <opencv2/opencv.hpp>

#include "../dataStructures/frameData.hpp"
#include "../dataStructures/pose.hpp"

class PoseCalculator
{
    public:
        PoseCalculator(){};
        ~PoseCalculator(){};

        virtual std::shared_ptr<Pose> calculate( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2 )=0;
};

std::shared_ptr<PoseCalculator> getPoseCalculator( std::string pose_calculation_method );




class NonePC : public PoseCalculator
{
    public:
        NonePC(){};
        ~NonePC(){};

        std::shared_ptr<Pose> calculate( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2 )override;
};


#endif