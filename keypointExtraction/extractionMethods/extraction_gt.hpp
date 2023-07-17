#ifndef extraction_gt_h
#define extraction_gt_h

#include <opencv2/opencv.hpp>

#include "../keypointExtraction.hpp"
#include "../../dataStructures/frameData.hpp"
#include "../../dataStructures/map3D.hpp"


class ORBNaiveBucketingGTExtractor : public Extractor
{
    private:
        cv::Ptr<cv::ORB> detector = cv::ORB::create(500);

    public:
        ORBNaiveBucketingGTExtractor(){};
        ~ORBNaiveBucketingGTExtractor(){};

        int extract(cv::Mat& img, 
                    std::shared_ptr<FrameData> frame, 
                    std::shared_ptr<Map3D> map_3d )override;
        void depthGTwBucketing( cv::Mat& img, 
                                std::shared_ptr<FrameData> frame, 
                                std::vector<cv::KeyPoint>& kpts, 
                                std::shared_ptr<Map3D> map_3d, 
                                int h_n_buckets, int w_n_buckets);
};

#endif