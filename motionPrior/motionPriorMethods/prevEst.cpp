#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>

#include "prevEst.hpp"
#include "../../util/util.hpp"


using std::vector;

PrevEstMP::PrevEstMP()
{
    YAML::Node config = YAML::LoadFile("config/pre_est_config.yaml");

    this->file_path = config["pose_pre_est.pose_folder"].as<std::string>();

    this->est_poses = readCSVFile(file_path, ' ');

}

void PrevEstMP::calculate( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2 )
{
    cv::Mat rel_T = this->relMotionPriorPrevEst(frame1, frame2);
    //frame1->setGlobalPose( T_wc );
    //std::cout << frame2->getGlobalPose() << std::endl;
    //cv::Mat rel_T = relTfromglobalTx2(T_wc, frame2->getGlobalPose());
    FrameData::registerGTRelPose(rel_T, frame1, frame2);
}

cv::Mat PrevEstMP::relMotionPriorPrevEst( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2 )
{
    std::string frame1_name, frame2_name, pose_name;
    vector<double> pose1, pose2;
    
    frame1_name = zeroPad(frame1->getImgId(), 6) + ".png";
    frame2_name = zeroPad(frame2->getImgId(), 6) + ".png";

    for (int i=0; i < this->est_poses.size(); i++)
    {
        pose_name = est_poses[i][0];
        if (pose_name == frame1_name)
        {
            for ( int j = 1; j < est_poses[i].size(); j++ )
            {
                pose1.push_back(std::stod(est_poses[i][j]));
            }
        }
        if (pose_name == frame2_name)
        {
            for ( int j = 1; j < est_poses[i].size(); j++ )
            {
                pose2.push_back(std::stod(est_poses[i][j]));
            }
        }
    }
    cv::Mat T1_wc = compileTMatrix(pose1);
    cv::Mat T2_wc = compileTMatrix(pose2);
    cv::Mat rel_T = relTfromglobalTx2(T1_wc, T2_wc);
    return rel_T;
}