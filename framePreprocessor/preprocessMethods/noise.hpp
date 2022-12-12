#ifndef noise_h
#define noise_h

#include "../framePreprocessor.hpp"

class Noise : public Preprocessor
{
    private:
        float mean = 0;
        float std = 20;
    public:
        Noise(){};
        ~Noise(){};

        void calculate( cv::Mat& img, std::shared_ptr<FrameData> frame )override;
};

#endif