#ifndef GJET_h
#define GJET_h

#include "../poseCalculation.hpp"
#include "../../dataStructures/frameData.hpp"
#include "../../dataStructures/pose.hpp"
#include "../../dataStructures/parametrization.hpp"


class GJET : public PoseCalculator    // 5-Point with Outlier Rejection Pose Calculator
{
    private:
        ParamID paramId = ParamID::STDPARAM;
    public:
        GJET(){};
        ~GJET(){};

        std::shared_ptr<Pose> calculate( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2 )override;
        void jointEpipolarOptimization( cv::Mat& F_matrix, std::vector<std::shared_ptr<KeyPoint2>>& matched_kpts1, std::vector<std::shared_ptr<KeyPoint2>>& matched_kpts2 );

        static double solveQuadraticFormForV( cv::Mat& A_k, cv::Mat& b_k, cv::Mat& c_k, cv::Mat& v_k );
        static cv::Mat solveKKT( cv::Mat& A, cv::Mat& g, cv::Mat& b, cv::Mat& h );
        static double epipolarConstrainedOptimization( const cv::Mat& F_matrix, const cv::Mat& A_d_k, const cv::Mat& x_k, const cv::Mat& y_k, cv::Mat& v_k_opt );


};

#endif