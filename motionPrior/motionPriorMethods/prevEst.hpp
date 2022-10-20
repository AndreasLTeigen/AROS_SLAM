#ifndef motionPrior_prevEval_h
#define motionPrior_prevEval_h

#include "../motionPrior.hpp"
#include "../../dataStructures/frameData.hpp"
#include "../../dataStructures/map3D.hpp"


// Loads a previous motion estimation from file to use as a motion prior
class PrevEstMP : public MotionPrior
{
    private:
        std::string file_path;
        std::vector<std::vector<std::string>> est_poses;
    public:
        PrevEstMP();
        ~PrevEstMP(){};

        void calculate( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2 )override;
        cv::Mat relMotionPriorPrevEst( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2 );
};

#endif