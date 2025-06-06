#include <opencv2/opencv.hpp>

#include "keypointExtraction.hpp"
#include "extractionMethods/orb.hpp"
#include "extractionMethods/orbOS.hpp"
// #include "extractionMethods/orb_test.hpp"
#include "extractionMethods/bucketing.hpp"
#include "extractionMethods/extraction_gt.hpp"
#include "extractionMethods/blockFeatures.hpp"

#include "../util/util.hpp"

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
    else if ( extractor_method == "orb_os" )
    {
        return std::make_shared<ORBOSExtractor>();
    }
    else if ( extractor_method == "orb_nb_gt" )
    {
        return std::make_shared<ORBNaiveBucketingGTExtractor>();
    }
    else if ( extractor_method == "blockFeature")
    {
        return std::make_shared<BlockFeatures>();
    }
    // else if (extractor_method == "orb_test")
    // {
    //     return std::make_shared<KptExtractor>();
    // }
    else
    {
        std::cerr << "Warning: Extraction method not found." << std::endl;
        return std::make_shared<NoneExtractor>();
    }
}


Extractor::Extractor()
{
    if (this->analysis_kpts_count)
    {
        clearFile(this->f_kpts_count);
    }
}

void Extractor::analysis(  cv::Mat &img, 
                                std::shared_ptr<FrameData> frame, 
                                std::shared_ptr<Map3D> map_3d )
{
    if (this->analysis_kpts_count)
    {
        writeInt2File(this->f_kpts_count, this->num_kpts_curr);
    }
}


int Extractor::getCurrKptNum()
{
    return this->num_kpts_curr;
}

int NoneExtractor::extract( cv::Mat& img, 
                            std::shared_ptr<FrameData> frame, 
                            std::shared_ptr<Map3D> map_3d )
{
    //std::cerr << "ERROR: KEYPOINT EXTRACTION ALGORITHM NOT IMPLEMENTED" << std::endl;
    return 0;
}