#include <opencv2/opencv.hpp>

#include"orb.hpp"

using std::string;
using std::vector;

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::high_resolution_clock;

using cv::Mat;
using cv::KeyPoint;
using cv::Ptr;

int ORBExtractor::extract(  cv::Mat& img, 
                            std::shared_ptr<FrameData> frame, 
                            std::shared_ptr<Map3D> map_3d )
{
    vector<cv::KeyPoint> kpts;
    Mat desc;

    auto detect_start = high_resolution_clock::now();

    //detector->detect( img, kpts );
    orb->detectAndCompute( img, cv::noArray(), kpts, desc );

    auto register_start = high_resolution_clock::now();

    frame->registerKeypoints(kpts, desc);

    auto full_end = high_resolution_clock::now();


    auto ms1 = duration_cast<milliseconds>(register_start-detect_start);
    auto ms3 = duration_cast<milliseconds>(full_end-register_start);

    std::cout << "Extract: " << ms1.count() << "ms" << std::endl;
    std::cout << "Registration: " << ms3.count() << "ms" << std::endl;
    
    return 0;
}