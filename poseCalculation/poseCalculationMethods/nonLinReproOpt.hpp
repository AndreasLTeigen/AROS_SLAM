#ifndef repro_opt_h
#define repro_opt_h

#include "../poseCalculation.hpp"
#include "../../dataStructures/frameData.hpp"
#include "../../dataStructures/pose.hpp"

class ReproOpt: public PoseCalculator
{
    private:
        std::shared_ptr<Pose> calculatePoseLinear( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2 );
        void kptTriangulationLinear( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2, cv::Mat& XYZ1 );
        double residual( cv::Mat& T_rel, cv::Mat& K1, cv::Mat& K2, cv::Mat& XYZ1, cv::Mat& uv1_1, cv::Mat& uv1_2 );

    public:
        ReproOpt(){};
        ~ReproOpt(){};

        std::shared_ptr<Pose> calculate( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2 )override;
};

#endif