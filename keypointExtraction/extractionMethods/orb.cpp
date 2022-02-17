#include <opencv2/opencv.hpp>

#include"orb.hpp"

using std::string;
using std::vector;

using cv::Mat;
using cv::KeyPoint;
using cv::Ptr;

void ORBExtractor::extract( cv::Mat& img, std::shared_ptr<FrameData> frame, std::shared_ptr<Map3D> map_3d )
{
    vector<cv::KeyPoint> kpts;
    Mat desc;

    detector->detect( img, kpts );
    descriptor->compute( img, kpts, desc);
    frame->registerKeypoints(kpts, desc);
}