#ifndef GJET_h
#define GJET_h

#include "../poseCalculation.hpp"
#include "../../dataStructures/frameData.hpp"
#include "../../dataStructures/pose.hpp"


class GJET : public PoseCalculator    // 5-Point with Outlier Rejection Pose Calculator
{
    private:
        double solveQuadraticFormForV( cv::Mat& A_k, cv::Mat& b_k, cv::Mat& c_k, cv::Mat& v_k );
        double epipolarConstrainedOptimization( cv::Mat& F_matrix, cv::Mat& A_d_k, cv::Mat& x_k, cv::Mat& y_k, cv::Mat& v_k_opt );
        void jointEpipolarOptimization( cv::Mat& F_matrix, std::vector<std::shared_ptr<KeyPoint2>>& matched_kpts1, std::vector<std::shared_ptr<KeyPoint2>>& matched_kpts2 );

    public:
        GJET(){};
        ~GJET(){};

        std::shared_ptr<Pose> calculate( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2 )override;
};

#endif