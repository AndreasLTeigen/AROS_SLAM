#ifndef blur_h
#define blur_h

#include "../framePreprocessor.hpp"

class Blur : public Preprocessor
{
    private:
        cv::Size kernel_size;
    public:
        Blur(const YAML::Node config);
        ~Blur(){};

        void calculate( cv::Mat& img )override;
};

#endif