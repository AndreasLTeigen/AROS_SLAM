#ifndef orbExtractor_h
#define orbExtractor_h

#include <opencv2/opencv.hpp>

#include "../keypointExtraction.hpp"
#include "../../dataStructures/frameData.hpp"
#include "../../dataStructures/map3D.hpp"

class ORBExtractor : public Extractor
{
    private:
        cv::Ptr<cv::ORB> orb = cv::ORB::create(500);
        //cv::Ptr<cv::ORB> descriptor = cv::ORB::create();

    public:
        ORBExtractor(){};
        ~ORBExtractor(){};

        void extract( cv::Mat& img, std::shared_ptr<FrameData> frame, std::shared_ptr<Map3D> map_3d )override;
};

#endif