#ifndef descrDistribExtractor_h
#define descrDistribExtractor_h

#include <opencv2/opencv.hpp>

#include "../keypointExtraction.hpp"
#include "../../dataStructures/frameData.hpp"
#include "../../dataStructures/map3D.hpp"

class DescrDistribExtractor : public Extractor
{
    private:
        int reg_size = 5;                                   // Size of local region of interest (around each keypoint)
        cv::Ptr<cv::ORB> orb = cv::ORB::create(500);
        //cv::Ptr<cv::ORB> descriptor = cv::ORB::create();

        std::vector<cv::KeyPoint> generateLocalRegionKpts( cv::KeyPoint kpt, int reg_size );

    public:
        DescrDistribExtractor(){};
        ~DescrDistribExtractor(){};

        void extract( cv::Mat& img, std::shared_ptr<FrameData> frame, std::shared_ptr<Map3D> map_3d )override;
};

#endif