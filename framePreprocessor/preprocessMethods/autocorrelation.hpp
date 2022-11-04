#ifndef autocorrelation_h
#define autocorrelation_h

#include "../framePreprocessor.hpp"

class Autocor : public Preprocessor
{
    private:
        cv::Size grid_resolution = cv::Size(1,1);
        cv::Size range = cv::Size(10,10);
        int step_size = 2;
    public:
        Autocor(){};
        ~Autocor(){};

        void calculate( cv::Mat& img, std::shared_ptr<FrameData> frame )override;
        int simpleAutocor( cv::Mat& img, cv::Mat& autocor );
        int autocorFilter( cv::Mat& img, cv::Mat& autocor );
        void applyHeatmap( cv::Mat& img, cv::Mat& autocov, std::vector<cv::Rect>& rect_vec );
};

#endif