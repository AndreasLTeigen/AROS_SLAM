#ifndef homomorphicFilter_h
#define homomorphicFilter_h

#include "../framePreprocessor.hpp"

class HighPassFilter {
public:
    virtual cv::Mat createFilter(int rows, int cols, float sigma, float alpha, float beta) const = 0;
    virtual ~HighPassFilter() = default;
};

class GaussianHighPassFilter: public HighPassFilter {
public:
    cv::Mat createFilter(int rows, int cols, float sigma, float alpha, float beta) const override;
};

class ButterworthHighPassFilter: public HighPassFilter {
private:
    int n;
public:
    ButterworthHighPassFilter(int n);
    cv::Mat createFilter(int rows, int cols, float sigma, float alpha, float beta) const override;
};

void homomorphicFilter(const cv::Mat& source, 
                       const cv::Mat& dest, 
                       float sigma, 
                       float alpha, 
                       float beta, 
                       HighPassFilter& hpf, 
                       int borderType = cv::BORDER_REPLICATE);
void dftShift(cv::InputOutputArray _out); 

class HomomorphicFiltering : public Preprocessor
{
    private:
        float alpha = 0.5;
        float beta = 2.0;
        float sigma = 15.0;
        HighPassFilter* hpf = new GaussianHighPassFilter();
    public:
        HomomorphicFiltering(){};
        ~HomomorphicFiltering(){};

        void calculate( cv::Mat& img )override;
};

#endif