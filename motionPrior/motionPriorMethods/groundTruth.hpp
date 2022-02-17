#ifndef motionPrior_gt_h
#define motionPrior_gt_h

#include "../motionPrior.hpp"
#include "../../dataStructures/frameData.hpp"
#include "../../dataStructures/map3D.hpp"


class GroundTruthMP : public MotionPrior
{
    public:
        GroundTruthMP(){};
        ~GroundTruthMP(){};

        void calculate( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2 )override;
        cv::Mat globalMotionPriorGT( std::shared_ptr<FrameData> frame1 );
};

#endif