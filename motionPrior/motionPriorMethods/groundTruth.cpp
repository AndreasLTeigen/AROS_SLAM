#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "groundTruth.hpp"
#include "../../util/util.hpp"

cv::Mat motionPriorGT(std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2)
{
    std::string timestamp_path = "/mnt/c/Users/and_t/Documents/AROS/Database/2021-08-17_SEQ1/vehicle0/cam0/camM0_timestamps.csv";
    std::string gt_poses_path = "/mnt/c/Users/and_t/Documents/AROS/Database/2021-08-17_SEQ1/vehicle0/cam0/camM0_poses/camM0_poses_transformation_matrix.csv";
    std::string image_prefix = "seq01_veh0_camM0_D-";


    // Retrieveing the timestamp corresponding with the image id
    long long int timestamp;
    std::string image_name, timestamp_name;
    std::vector<std::vector<std::string>> timestamps = readCSVFile(timestamp_path);
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
    std::vector<std::vector<std::string>> poses_gt = readCSVFile(gt_poses_path);
    for ( int i = 1; i < poses_gt.size(); i++ )
    {
        if ( timestamp == std::stoll(poses_gt[i][0]) )
        {
            for ( int j = 1; j < poses_gt[i].size(); j++ )
            {   
                pose_gt.push_back(std::stod(poses_gt[i][j]));
            }
            break;
        }
    }

    cv::Mat T = compileTMatrix(pose_gt);

    cv::Mat rel_T = relTfromglobalTx2(T, frame2->getGlobalPose());
    
    return rel_T;
}