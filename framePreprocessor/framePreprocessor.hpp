#ifndef framePreprocessor_h
#define framePreprocessor_h

#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>

#include "../dataStructures/frameData.hpp"

class Preprocessor
{
    public:
        Preprocessor(){};
        ~Preprocessor(){};

        virtual void calculate( cv::Mat& img )=0;
};

std::shared_ptr<Preprocessor> getPreprocessor( YAML::Node config );




class NoneProcessor : public Preprocessor
{
    public:
        NoneProcessor();
        ~NoneProcessor(){};

        void calculate( cv::Mat& img )override;
};

#endif