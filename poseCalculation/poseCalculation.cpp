#include <math.h>
#include <opencv2/opencv.hpp>

#include "poseCalculation.hpp"
#include "poseCalculationMethods/point5OutlierRejection.hpp"
#include "poseCalculationMethods/nonLinReproOpt.hpp"
#include "poseCalculationMethods/GJET2.hpp"
#include "poseCalculationMethods/copyMotionPrior.hpp"
#include "poseCalculationMethods/blockShift.hpp"

#include "../util/util.hpp"


std::shared_ptr<PoseCalculator> getPoseCalculator( std::string pose_calculation_method )
{
    if ( pose_calculation_method == "5-point" )
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
    else if ( pose_calculation_method == "motionPrior" )
    {
        return std::make_shared<CopyMPPC>();
    }
    else if ( pose_calculation_method == "blockShift" )
    {
        return std::make_shared<BlockShift>();
    }
    else
    {
        std::cerr << "Warning: Pose calculation method not found." << std::endl;
        return std::make_shared<NonePC>();
    }
}

ParamID getParametrization( std::string parametrization_method )
{
    if ( parametrization_method == "std" )
    {
        std::cout << "Method: Using standard parametrization." << std::endl;
        return ParamID::STDPARAM;
    }
    else if ( parametrization_method == "lie")
    {
        std::cout << "Method: Using angle axis parametrization." << std::endl;
        return ParamID::LIEPARAM;
    }
    else
    {
        std::cerr << "Warning: Parametrization method not found." << std::endl;
        return ParamID::NONE;
    }
}

PoseCalculator::PoseCalculator()
{
    if (this->analysis_reprojection_error)
    {
        clearFile(this->f_rep_error);
    }
    if (this->analysis_outlier_count)
    {
        clearFile(this->f_outlier_count);
        clearFile(this->f_inlier_count);
    }
    if (this->analysis_hamming_distance)
    {
        clearFile(this->f_hamming_distance);
    }
}

void PoseCalculator::analysis(  std::shared_ptr<FrameData> frame1, 
                                std::shared_ptr<FrameData> frame2,
                                cv::Mat& img )
{
    if (this->analysis_reprojection_error)
    {
        /* Writes the reprojection error of all remaning matches to file */
        cv::Mat E_matrix, F_matrix, K1, K2, reprojection_error;
        std::shared_ptr<Pose> rel_pose;

        rel_pose = frame1->getRelPose( frame2 );
        E_matrix = rel_pose->getEMatrix();
        F_matrix = fundamentalFromEssential(E_matrix, 
                                        frame1->getKMatrix(), 
                                        frame2->getKMatrix() ).t();
        K1 = frame1->getKMatrix();
        K2 = frame2->getKMatrix();

        cv::Mat frame1_points, frame2_points;
        compileMatchedPointCoords(  frame1, frame2, 
                                    frame1_points, frame2_points);

        reprojection_error = cv::Mat::zeros(1,frame1_points.cols, CV_64F);

        computeReprojectionError(   F_matrix, K1, K2, 
                                    frame1_points, frame2_points,
                                    reprojection_error );
        writeMat2File(this->f_rep_error, reprojection_error);
    }

    if (this->analysis_outlier_count)
    {
        writeInt2File(this->f_outlier_count, this->num_outliers);
        writeInt2File(this->f_inlier_count, this->num_inliers);
    }
    
    if (this->analysis_hamming_distance)
    {
        double descriptor_distance;
        std::shared_ptr<Match> match;
        std::vector<std::shared_ptr<KeyPoint2>> matched_kpts 
                    = frame1->getMatchedKeypoints(frame2->getFrameNr());

        for (std::shared_ptr<KeyPoint2> kpt : matched_kpts)
        {
            match = kpt->getHighestConfidenceMatch(frame2->getFrameNr());
            descriptor_distance = match->getDescrDistance();
            writeInt2File(this->f_hamming_distance, int(descriptor_distance));
        }
        writeString2File(this->f_hamming_distance,""); //Break line
    }
}

bool PoseCalculator::isStationaryFrame( std::vector<cv::Point> pts1, 
                                        std::vector<cv::Point> pts2)
{
    /*
    Determines if the frame is stationary (no keypoint movement) compared
    to the last frame.

    Argumnets:
        pts1:               Points in current frame.
        pts2:               Points in previous frame.
    Returns:
        isStationaryFrame:  True if average keypoint movement is less than 
                            1 pixel.
    */

    double x1,y1, x2, y2, mean;

    int N = pts1.size();
    cv::Mat distance = cv::Mat::zeros(1, N, CV_64F);
    for (int i = 0; i < N; ++i)
    {
        x1 = double(pts1[i].x);
        y1 = double(pts1[i].y);
        x2 = double(pts2[i].x);
        y2 = double(pts2[i].y);
        distance.at<double>(0,i) = 
                    std::sqrt((x1-x2) * (x1-x2) + (y1-y2)*(y1-y2));
    }
    mean = cv::mean(distance)[0];
    
    if (mean < this->stationarity_threshold)
    {
        return true;
    }
    else
    {
        return false;
    }
}


int NonePC::calculate( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2, cv::Mat& img )
{
    //std::cerr << "ERROR: POSE CALCULATION ALGORITHM NOT IMPLEMENTED" << std::endl;
    return 1;
}