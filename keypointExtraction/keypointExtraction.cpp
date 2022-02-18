#include <opencv2/opencv.hpp>

#include "keypointExtraction.hpp"
#include "extractionMethods/orb.hpp"
#include "extractionMethods/bucketing.hpp"
#include "extractionMethods/extraction_gt.hpp"

using std::string;
using std::vector;

using cv::Mat;
using cv::KeyPoint;
using cv::Ptr;





std::shared_ptr<Extractor> getExtractor( string extractor_method )
{
    if ( extractor_method == "orb" )
    {
        return std::make_shared<ORBExtractor>();
    }
    else if ( extractor_method == "orb_nb_gt" )
    {
        return std::make_shared<ORBNaiveBucketingGTExtractor>();
    }
    else
    {
        std::cerr << "ERROR: EXTRACTION METHOD NOT FOUND" << std::endl;
        return nullptr;
    }
}




void NoneExtractor::extract( cv::Mat& img, std::shared_ptr<FrameData> frame, std::shared_ptr<Map3D> map_3d )
{
    std::cerr << "ERROR: KEYPOINT EXTRACTION ALGORITHM NOT IMPLEMENTED" << std::endl;
}