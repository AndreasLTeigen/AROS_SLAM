#include <string>
#include <vector>
#include <memory>
#include <iomanip>
#include <iostream>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>

#include "depthGT.hpp"
#include "../../util/util.hpp"


void depthGTMPReg( std::shared_ptr<FrameData> frame1, std::shared_ptr<Map3D> map_3d)
{
    YAML::Node config = YAML::LoadFile("config/gt_config.yaml");

    double max_depth_m = config["depth_gt.max_depth_m"].as<double>();
    int max_depth_p = config["depth_gt.max_depth_p"].as<int>();
    std::string depth_gt_folder = config["depth_gt.folder"].as<std::string>();
    std::string depth_gt_prefix = config["depth_gt.prefix"].as<std::string>();
    std::string depth_gt_image = depth_gt_prefix + zeroPad(frame1->getImgId(), 8);
    std::string depth_gt_file_format = config["depth_gt.file_format"].as<std::string>();
    std::string depth_gt_path = depth_gt_folder + depth_gt_image + depth_gt_file_format;

    cv::Mat depth_gt = cv::imread(depth_gt_path, cv::IMREAD_ANYDEPTH);
    if(depth_gt.empty())
    {
        std::cout << "ERROR: COULD NOT READ THE DEPTH GROUND TRUTH: " << depth_gt_path << std::endl;
    }
    

    int depth_p;    // Depth pixel value
    double depth_m;  // Depth meter value
    cv::Mat xy1, XYZ;
    cv::Mat T1 = frame1->getGlobalPose();
    std::vector<std::shared_ptr<KeyPoint2>> kpts = frame1->getKeypoints();

    for (std::shared_ptr<KeyPoint2> kpt : kpts )
    {
        if ( kpt->getMapPoint() == nullptr )
        {
            depth_p = int( depth_gt.at<ushort>(int(kpt->getCoordY()), int(kpt->getCoordX())) );
            depth_m = depth_p*max_depth_m /  max_depth_p;
            xy1 = xyToxy1( kpt->getCoordX(), kpt->getCoordY() );
            XYZ = dilateKptWDepth(xy1, depth_m, T1, frame1->getKMatrix());
            map_3d->createMapPoint(XYZ, cv::Mat::zeros(3, 1, CV_64F), kpt, T1);
        }
    }
    //std::cout << depth_gt.size() << std::endl;
    //std::cout << depth_gt << std::endl;
    //depth_p = int(depth_gt.at<ushort>(719, 1279));
    //std::cout << depth_p << std::endl;
    //std::cout << depth_p*max_depth_m /  max_depth_p << std::endl;
}