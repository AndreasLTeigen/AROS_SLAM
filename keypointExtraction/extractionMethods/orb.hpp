#ifndef orbExtractor_h
#define orbExtractor_h

#include <opencv2/opencv.hpp>

#include "../keypointExtraction.hpp"
#include "../../dataStructures/frameData.hpp"
#include "../../dataStructures/map3D.hpp"

class ORBExtractor : public Extractor
{
    private:
        bool non_rot_desc = true;
        int nfeatures = 500;
        float scaleFactor = 1.2f;
        int nlevels = 8;
        cv::Ptr<cv::ORB> orb = cv::ORB::create(nfeatures, scaleFactor, nlevels);

    public:
        ORBExtractor(){};
        ~ORBExtractor(){};

        int extract(    cv::Mat& img, 
                        std::shared_ptr<FrameData> frame, 
                        std::shared_ptr<Map3D> map_3d )override;
};

#endif