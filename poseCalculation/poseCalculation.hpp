#ifndef poseCalculation_h
#define poseCalculation_h

#include <opencv2/opencv.hpp>

#include "../dataStructures/frameData.hpp"
#include "../dataStructures/pose.hpp"


// P5OR - 5 point + outlier removal

enum class PoseCalculator {P5OR, MP, NONE};

PoseCalculator getRelativePoseCalculationMethod( std::string pose_calculation_method );

std::shared_ptr<Pose> calculateRelativePose(std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2, cv::Mat K_matrix, PoseCalculator pose_calculation_type);

#endif