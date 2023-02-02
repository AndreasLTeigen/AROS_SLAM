#ifndef blur_h
#define blur_h

#include "../framePreprocessor.hpp"

class Blur : public Preprocessor
{
    private:
        cv::Size kernel_size = cv::Size(5,5);
    public:
        Blur(){};
        ~Blur(){};

        void calculate( cv::Mat& img )override;
};

#endif