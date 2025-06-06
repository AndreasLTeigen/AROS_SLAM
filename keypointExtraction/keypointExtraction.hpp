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

        // Analysis toggle
        bool analysis_kpts_count = true;

        // Analysis filenames
        std::string f_kpts_count = "output/analysis/kpts_count.txt";

        
        Extractor();
        ~Extractor(){};

        virtual int extract(cv::Mat& img, 
                            std::shared_ptr<FrameData> frame, 
                            std::shared_ptr<Map3D> map_3d )=0;
        void analysis(  cv::Mat& img, 
                        std::shared_ptr<FrameData> frame, 
                        std::shared_ptr<Map3D> map_3d );

        int getCurrKptNum();
};

std::shared_ptr<Extractor> getExtractor( std::string extractor_method );




class NoneExtractor : public Extractor
{
    public:
        NoneExtractor(){};
        ~NoneExtractor(){};

        int extract(cv::Mat& img, 
                    std::shared_ptr<FrameData> frame, 
                    std::shared_ptr<Map3D> map_3d )override;
};

#endif