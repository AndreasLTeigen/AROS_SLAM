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

        int calculate( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2, cv::Mat& img )override;
        void resetKptMatches( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2 );
};

#endif