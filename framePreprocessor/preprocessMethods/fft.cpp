#include <opencv2/opencv.hpp>

#include "fft.hpp"
#include "../../util/util.hpp"


using cv::Mat;

void FFT::calculate( cv::Mat& img, std::shared_ptr<FrameData> frame )
{
    //std::cout << "Performing fourier transform..." << std::endl;
    Mat out;

    img.convertTo(out, CV_32FC1, 1.0/255.0);
    cv::dft(out, out);



    //this->fft(img, out);
    //this->fftshift( out );

    //normalize(out, out, 0, 1, cv::NORM_MINMAX); // Transform the matrix with float values into a
                                            // viewable image form (float between values 0 and 1).

    //this->enchanceFreq(out, out);
    std::cout << out.size() << std::endl;
    std::cout << out.channels() << std::endl;

    cv::dft(out, img, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);
    std::cout << img.size() << std::endl;
    std::cout << img.channels() << std::endl;

    print(img);
    img.convertTo(img, CV_8UC1);

    std::cout << img.size() << std::endl;
    std::cout << img.channels() << std::endl;

    //cv::imshow("Input Image"       , img);    // Show the result
    cv::imshow("spectrum magnitude", img);
    cv::waitKey();
}

cv::Size FFT::getOptimalSize( cv::Mat& img )
{
    int nrows, ncols;

    nrows = cv::getOptimalDFTSize(img.rows);
    ncols = cv::getOptimalDFTSize(img.cols);

    return cv::Size(ncols, nrows);
}

void FFT::fft( cv::Mat& src, cv::Mat& dst )
{
    int right, bottom;
    Mat src_pad;
    cv::Size dftSize = this->getOptimalSize(src);

    right = dftSize.width - src.cols;
    bottom = dftSize.height - src.rows;

    cv::copyMakeBorder(src, src_pad, 0,bottom,0,right,cv::BORDER_CONSTANT, cv::Scalar::all(0));
    src_pad.convertTo(src_pad, CV_32FC1, 1.0/255.0);

    // Borrowed from opencv c++ tutorial on fourier transform.
    Mat planes[] = {cv::Mat_<float>(src_pad), Mat::zeros(src_pad.size(), CV_32F)};
    Mat complexI;
    cv::merge(planes, 2, complexI);                 // Add to the expanded another plane with zeros

    cv::dft( complexI, complexI );              // this way the result may fit in the source matrix
    
    // compute the magnitude and switch to logarithmic scale
    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    split(complexI, planes);                    // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    magnitude(planes[0], planes[1], planes[0]); // planes[0] = magnitude
    Mat magI = planes[0];

    magI += cv::Scalar::all(1);                     // switch to logarithmic scale
    log(magI, magI);

    dst = magI;
}

void FFT::fftshift( cv::Mat& pow_spec )
{
    // crop the spectrum, if it has an odd number of rows or columns
    pow_spec = pow_spec(cv::Rect(0, 0, pow_spec.cols & -2, pow_spec.rows & -2));
    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = pow_spec.cols/2;
    int cy = pow_spec.rows/2;

    Mat q0(pow_spec, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(pow_spec, cv::Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(pow_spec, cv::Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(pow_spec, cv::Rect(cx, cy, cx, cy)); // Bottom-Right
    
    Mat tmp;                            // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);                     // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
}


void FFT::enchanceFreq( cv::Mat& src, cv::Mat& dst, float value, float threshold )
{
    dst = src.clone();
    cv::Mat mask = (dst > threshold);
    dst.setTo(value, mask);

}