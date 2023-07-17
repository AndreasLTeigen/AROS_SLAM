#ifndef CopyMPPC_h
#define CopyMPPC_h


#include "../poseCalculation.hpp"
#include "../../dataStructures/frameData.hpp"
#include "../../dataStructures/pose.hpp"


class CopyMPPC : public PoseCalculator    // 5-Point with Outlier Rejection Pose Calculator
{
    private:
        bool remove_outliers = true;
        double inlier_threshold = 2.0;

    public:
        CopyMPPC(){};
        ~CopyMPPC(){};

        int calculate(  std::shared_ptr<FrameData> frame1, 
                        std::shared_ptr<FrameData> frame2, 
                        cv::Mat& img )override;
};

#endif