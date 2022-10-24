#ifndef blockShift_h
#define blockShift_h

#include "../poseCalculation.hpp"
#include "../../dataStructures/frameData.hpp"
#include "../../dataStructures/pose.hpp"

class BlockShift : public PoseCalculator
{
    public:
        BlockShift(){};
        ~BlockShift(){};

        std::shared_ptr<Pose> calculate( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2, cv::Mat& img )override;
        void analysis( cv::Mat &img_disp, std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2 )override;
};

#endif