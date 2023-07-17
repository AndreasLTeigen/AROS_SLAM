#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>

#include "groundTruth.hpp"
#include "../../util/util.hpp"

GroundTruthMP::GroundTruthMP()
{
    // ------ KITTI ------
    YAML::Node config = YAML::LoadFile("config/data_conf/kitti.yaml");
    std::string path_gt_pose = config["Gt.pose_folder"].as<std::string>();
    this->poses_gt = readCSVFile(path_gt_pose, ' ');
}

void GroundTruthMP::calculate( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2 )
{
    cv::Mat rel_T = this->globalMotionPriorGTKitti(frame1, frame2);
    //frame1->setGlobalPose( T_wc );
    //cv::Mat rel_T = relTfromglobalTx2(T_wc, frame2->getGlobalPose());
    FrameData::registerGTRelPose(rel_T, frame1, frame2);
}

cv::Mat GroundTruthMP::globalMotionPriorGTKitti(std::shared_ptr<FrameData> frame1,
                                                std::shared_ptr<FrameData> frame2 )
{
    int img1_id = frame1->getImgId();
    int img2_id = frame2->getImgId();
    cv::Mat T_wc1, T_wc2;
    std::vector<double> gt_pose1, gt_pose2;

    for (int j = 0; j < this->poses_gt[img1_id].size(); ++j)
    {
        gt_pose1.push_back(std::stod(poses_gt[img1_id][j]));
        gt_pose2.push_back(std::stod(poses_gt[img2_id][j]));
    }
    T_wc1 = compileTMatrix(gt_pose1);
    T_wc2 = compileTMatrix(gt_pose2);

    frame1->setGlobalPose( T_wc1 );
    frame2->setGlobalPose( T_wc2 );
    return relTfromglobalTx2(T_wc1, T_wc2);
}

cv::Mat GroundTruthMP::globalMotionPriorGTVaros( std::shared_ptr<FrameData> frame1 )
{
    YAML::Node config = YAML::LoadFile("config/gt_config.yaml");

    std::string timestamp_path = config["pose_gt.timestamp_folder"].as<std::string>();
    std::string gt_poses_path = config["pose_gt.pose_folder"].as<std::string>();
    std::string image_prefix = config["pose_gt.prefix"].as<std::string>();


    // Retrieveing the timestamp corresponding with the image id
    long long int timestamp;
    std::string image_name, timestamp_name;
    std::vector<std::vector<std::string>> timestamps = readCSVFile(timestamp_path, ',');
    for(int i=1;i<timestamps.size();i++)
	{
        image_name = image_prefix + zeroPad(frame1->getImgId(), 8);
        timestamp_name = timestamps[i][4].substr(0, image_name.length());

        if (image_name == timestamp_name)
        {
            timestamp = std::stoll(timestamps[i][0]);
            break;
        }
	}
    
    // Retrieving the ground truth pose based on the timestamp
    std::vector<double> pose_gt;
    std::vector<std::vector<std::string>> poses_gt = readCSVFile(gt_poses_path, ',');
    for ( int i = 1; i < poses_gt.size(); i++ )
    {
        if ( timestamp == std::stoll(poses_gt[i][0]) )
        {
            std::cout << "Read pose: " << std::endl;
            for ( int j = 1; j < poses_gt[i].size(); j++ )
            {   
                pose_gt.push_back(std::stod(poses_gt[i][j]));
                std::cout << std::stod(poses_gt[i][j]) << "     ";
            }
            std::cout << std::endl;
            break;
        }
    }

    cv::Mat T_wc = compileTMatrix(pose_gt);
    
    return T_wc;
}