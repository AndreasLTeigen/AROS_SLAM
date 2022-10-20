#include <opencv2/opencv.hpp>

#include "poseCalculation.hpp"
#include "poseCalculationMethods/point5OutlierRejection.hpp"
#include "poseCalculationMethods/nonLinReproOpt.hpp"
#include "poseCalculationMethods/GJET.hpp"
#include "poseCalculationMethods/copyMotionPrior.hpp"


std::shared_ptr<PoseCalculator> getPoseCalculator( std::string pose_calculation_method )
{
    if ( pose_calculation_method == "5-point + outlier removal" )
    {
        return std::make_shared<P5ORPC>();
    }
    else if ( pose_calculation_method == "reproOpt")
    {
        return std::make_shared<ReproOpt>();
    }
    else if ( pose_calculation_method == "G_JET")
    {
        return std::make_shared<GJET>();
    }
    else if ( pose_calculation_method == "motion prior" )
    {
        return std::make_shared<CopyMPPC>();
    }
    else
    {
        std::cerr << "ERROR: POSE CALCULATION METHOD NOT FOUND" << std::endl;
        return std::make_shared<NonePC>();
    }
}

ParamID getParametrization( std::string parametrization_method )
{
    if ( parametrization_method == "std" )
    {
        std::cout << "METHOD: USING STANDARD PARAMETRIZATION" << std::endl;
        return ParamID::STDPARAM;
    }
    else if ( parametrization_method == "lie")
    {
        std::cout << "METHOD: USING ANGLE AXIS PARAMETRIZATION" << std::endl;
        return ParamID::LIEPARAM;
    }
    else
    {
        std::cerr << "ERROR: PARAMETRIZATION METHOD NOT FOUND" << std::endl;
        return ParamID::NONE;
    }
}




std::shared_ptr<Pose> NonePC::calculate( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2, cv::Mat& img )
{
    std::cerr << "ERROR: POSE CALCULATION ALGORITHM NOT IMPLEMENTED" << std::endl;
    return nullptr;
}

void NonePC::analysis( cv::Mat &img_disp, std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2 )
{
    std::cerr << "ERROR: POSE CALCULATION ANALYSIS ALGORITHM NOT IMPLEMENTED" << std::endl;
}