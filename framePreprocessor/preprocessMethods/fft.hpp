#ifndef fft_h
#define fft_h

#include "../framePreprocessor.hpp"

class FFT : public Preprocessor
{
    public:
        FFT(){};
        ~FFT(){};

        void calculate( cv::Mat& img, std::shared_ptr<FrameData> frame )override;

        cv::Size getOptimalSize( cv::Mat& img );
        void fft( cv::Mat& src, cv::Mat& dst );
        void fftshift( cv::Mat& pow_spec );

        void enchanceFreq( cv::Mat& source, cv::Mat& dst, float value = 1, float threshold = 0.3 );
};

#endif