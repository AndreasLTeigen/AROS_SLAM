#ifndef framePreprocessor_h
#define framePreprocessor_h

#include <opencv2/opencv.hpp>

#include "../dataStructures/frameData.hpp"

class Preprocessor
{
    public:
        Preprocessor(){};
        ~Preprocessor(){};

        virtual void calculate( cv::Mat& img, std::shared_ptr<FrameData> frame )=0;
};

std::shared_ptr<Preprocessor> getPreprocessor( std::string preprocessor_method );




class NoneProcessor : public Preprocessor
{
    public:
        NoneProcessor(){};
        ~NoneProcessor(){};

        void calculate( cv::Mat& img, std::shared_ptr<FrameData> frame )override;
};

#endif