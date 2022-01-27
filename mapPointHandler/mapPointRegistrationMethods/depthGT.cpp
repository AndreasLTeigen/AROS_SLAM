#include <string>
#include <vector>
#include <memory>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "depthGT.hpp"
#include "../../util/util.hpp"


void depthGTMPReg( std::shared_ptr<FrameData> frame1, std::shared_ptr<Map3D> map_3d)
{
    double max_depth_m = 25;
    int max_depth_p = 65535;
    std::string depth_gt_folder = "/mnt/c/Users/and_t/Documents/AROS/Database/2021-08-17_SEQ1/vehicle0/cam0/D/";
    std::string depth_gt_image = "seq01_veh0_camM0_D-" + zeroPad(frame1->getImgId(), 8);
    std::string depth_gt_file_format = ".png";
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
        depth_p = int( depth_gt.at<ushort>(int(kpt->getCoordY()), int(kpt->getCoordX())) );
        depth_m = depth_p*max_depth_m /  max_depth_p;
        xy1 = xyToxy1( kpt->getCoordX(), kpt->getCoordY() );
        XYZ = dilateKptWDepth(xy1, depth_m, T1, frame1->getKMatrix());
        map_3d->createMapPoint(XYZ, cv::Mat::zeros(3, 1, CV_64F), kpt, T1);
    }
    //std::cout << depth_gt.size() << std::endl;
    //std::cout << depth_gt << std::endl;
    //depth = int(depth_gt.at<ushort>(719, 1279));
    //std::cout << depth << std::endl;
    //std::cout << depth*max_depth_m /  max_depth_p << std::endl;
}