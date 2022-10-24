#ifndef blockFeatures_h
#define blockFeatures_h

#include "../keypointExtraction.hpp"

class BlockFeatures : public Extractor
{
    private:
        cv::Size grid_size = cv::Size(10,10);
    public:
        BlockFeatures(){};
        ~BlockFeatures(){};
    
        void extract( cv::Mat& img, std::shared_ptr<FrameData> frame, std::shared_ptr<Map3D> map_3d )override;
};


#endif