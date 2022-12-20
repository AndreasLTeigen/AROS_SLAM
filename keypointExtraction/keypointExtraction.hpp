#ifndef findKeypoints_h
#define findKeypoints_h

#include <opencv2/opencv.hpp>

#include "../dataStructures/frameData.hpp"
#include "../dataStructures/map3D.hpp"

class Extractor
{
    public:
        // Logging parameters
        int num_kpts_curr = -1;

        
        Extractor(){};
        ~Extractor(){};

        virtual void extract( cv::Mat& img, std::shared_ptr<FrameData> frame, std::shared_ptr<Map3D> map_3d )=0;

        int getCurrKptNum();
};

std::shared_ptr<Extractor> getExtractor( std::string extractor_method );




class NoneExtractor : public Extractor
{
    public:
        NoneExtractor(){};
        ~NoneExtractor(){};

        void extract( cv::Mat& img, std::shared_ptr<FrameData> frame, std::shared_ptr<Map3D> map_3d )override;
};

#endif