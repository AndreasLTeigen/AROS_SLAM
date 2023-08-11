#ifndef poseCalculation_h
#define poseCalculation_h

#include <opencv2/opencv.hpp>

#include "../dataStructures/frameData.hpp"
#include "../dataStructures/pose.hpp"
#include "../dataStructures/parametrization.hpp"

class PoseCalculator
{
    private:
        double stationarity_threshold = 12.5;
    public:
        // Log variables
        int num_outliers = -1;
        int num_inliers = -1;

        // Analysis toggles
        bool analysis_reprojection_error = true;
        bool analysis_outlier_count = true;
        bool analysis_hamming_distance = true;

        // Analysis filenames
        std::string f_rep_error = "output/analysis/rep_err.txt";
        std::string f_outlier_count = "output/analysis/outlier_count.txt";
        std::string f_inlier_count = "output/analysis/inlier_count.txt";
        std::string f_hamming_distance = "output/analysis/hamming_distance.txt";

        
        PoseCalculator();
        ~PoseCalculator(){};

        virtual int calculate( 
                    std::shared_ptr<FrameData> frame1, 
                    std::shared_ptr<FrameData> frame2, 
                    cv::Mat& img )=0;
        virtual void analysis(  std::shared_ptr<FrameData> frame1, 
                                std::shared_ptr<FrameData> frame2,
                                cv::Mat& img );
        bool isStationaryFrame( std::vector<cv::Point> pts1, 
                                std::vector<cv::Point> pts2);
};

std::shared_ptr<PoseCalculator> getPoseCalculator( std::string pose_calculation_method );

ParamID getParametrization( std::string parametrization_method );




class NonePC : public PoseCalculator
{
    public:
        NonePC(){};
        ~NonePC(){};

        int calculate( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2, cv::Mat& img )override;
};


#endif