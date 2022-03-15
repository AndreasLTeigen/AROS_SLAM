#include <opencv2/opencv.hpp>

#include"descriptorDistribution.hpp"

using std::string;
using std::vector;

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::high_resolution_clock;

using cv::Mat;
using cv::KeyPoint;
using cv::Ptr;

std::vector<cv::KeyPoint> DescrDistribExtractor::generateLocalRegionKpts( cv::KeyPoint kpt, int reg_size )
{
    float ref_x, ref_y, x, y, size;
    vector<cv::KeyPoint> local_kpts; //(reg_size*reg_size);

    ref_x = kpt.pt.x - reg_size/2; 
    ref_y = kpt.pt.y - reg_size/2;
    
    for ( int row_i = 0; row_i < reg_size; ++row_i )
    {
        y = ref_y + row_i;
        for ( int col_j = 0; col_j < reg_size; ++col_j )
        {
            x = ref_x + col_j;

            size = kpt.size;
            local_kpts.push_back(cv::KeyPoint(x, y, size));
        }
    }
    
    return local_kpts;
}

void DescrDistribExtractor::extract( cv::Mat& img, std::shared_ptr<FrameData> frame, std::shared_ptr<Map3D> map_3d )
{
    vector<cv::KeyPoint> kpts;
    Mat desc;

    auto detect_start = high_resolution_clock::now();

    orb->detect( img, kpts );

    for ( cv::KeyPoint kpt : kpts )
    {
        this->generateLocalRegionKpts(kpt, this->reg_size);
    }


    orb->compute( img, kpts, desc );

    auto register_start = high_resolution_clock::now();

    frame->registerKeypoints( kpts, desc );

    auto full_end = high_resolution_clock::now();


    auto ms1 = duration_cast<milliseconds>(register_start-detect_start);
    auto ms3 = duration_cast<milliseconds>(full_end-register_start);

    std::cout << "Extract: " << ms1.count() << "ms" << std::endl;
    std::cout << "Registration: " << ms3.count() << "ms" << std::endl;
}