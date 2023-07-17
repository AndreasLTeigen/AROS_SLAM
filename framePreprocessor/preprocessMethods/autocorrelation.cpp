#include <opencv2/opencv.hpp>

#include "autocorrelation.hpp"
#include "../../util/util.hpp"

using std::vector;
using std::shared_ptr;

void Autocor::calculate( cv::Mat& img )
{
    //img = cv::Mat::zeros(img.rows, img.cols, CV_8UC1) + 1;
    //img = cv::Mat::zeros(11, 16, CV_8UC1) + 1;
    //cv::randu(img, cv::Scalar(0), cv::Scalar(255));
    cv::Size rect_size = cv::Size(img.cols / grid_resolution.width, img.rows / grid_resolution.height);
    cv::Size rect_rest = cv::Size(img.cols % grid_resolution.width, img.rows % grid_resolution.height);

    int rect_x = 0, rect_y = 0, rect_w = 0, rect_h = 0;
    cv::Rect rect;
    cv::Mat autocor_mat, autocor;
    vector<cv::Rect> rect_vec;
    cv::Mat img_section;
    for ( int i = 0; i < this->grid_resolution.height; ++i )
    {
        rect_x = 0, rect_w = 0;
        rect_y = rect_y + rect_h;
        rect_h = rect_size.height;
        if ( i < rect_rest.height )
        {
            rect_h += 1;
        }
        for ( int j = 0; j < this->grid_resolution.width; ++j )
        {
            rect_x = rect_x + rect_w;
            rect_w = rect_size.width;
            if ( j < rect_rest.width )
            {
                rect_w += 1;
            }
            if (rect_w == 0 || rect_h == 0)
            {
                continue;
            }
            rect = cv::Rect( rect_x, rect_y, rect_w, rect_h );
            //std::cout << rect << std::endl;
            img_section = img(rect);
            this->simpleAutocor(img_section, autocor);
            //this->autocorFilter(img_section, autocor);

            autocor_mat.push_back(autocor.t());
            rect_vec.push_back(rect);
        }
    }
    //std::cout << autocor_mat << std::endl;
    //this->applyHeatmap(img, autocor_mat, rect_vec);
}

int Autocor::autocorFilter( cv::Mat& img, cv::Mat& autocor )
{
    // WARNING: UNFINISHED
    int N;
    int M = this->range.width/this->step_size;
    cv::Mat kernel, dst, img_64f;
    vector<double> autocor_vec(M);
    img.convertTo(img_64f, CV_64F);

    for ( int step_m = 0; step_m < M; ++step_m )
    {
        kernel = cv::Mat::ones(2*step_m+1, 2*step_m+1, CV_64F);
        N = 8*step_m; // Wrong
        cv::filter2D(img_64f, dst, -1, kernel);
        autocor_vec[step_m] = cv::sum(cv::mean(dst))[0];
    }
    autocor = cv::Mat(autocor_vec, CV_64F);
    return 0;
}

int Autocor::simpleAutocor( cv::Mat& img, cv::Mat& autocor )
{
    int M = this->range.width/this->step_size;
    double autocor_val;
    cv::Mat accum_mat = cv::Mat::zeros(1, M, CV_64F);

    for ( int step_m = 0; step_m < M; ++step_m )
    {
        for ( int row_i = 0; row_i < img.rows; ++row_i )
        {
            for ( int col_j = 0; col_j < img.cols - step_m; ++col_j )
            {

                autocor_val = double(img.at<uchar>(row_i, col_j)) * double(img.at<uchar>(row_i, col_j + step_m));
                accum_mat.at<double>(0, step_m) += autocor_val;
            }
        }
    }
    autocor = (accum_mat / (img.rows * img.cols)).t();

    //double mean = cv::sum(cv::mean(img))[0];
    //autocor = (autocor - mean*mean);
    autocor = autocor / autocor.at<double>(0,0);
    return 0;
}

void Autocor::applyHeatmap( cv::Mat& img, cv::Mat& autocov, vector<cv::Rect>& rect_vec )
{
    cv::Mat autocov_deriv = ( autocov.col(1) - autocov.col(int(this->range.width/this->step_size - 1)) ) \
                                / int(this->range.width/this->step_size);
    //std::cout << autocov << std::endl;
    //std::cout << autocov_deriv << std::endl;
    
    cv::Rect rect;
    cv::Mat canvas = cv::Mat::zeros(img.size(), CV_64F);
    cv::Size rect_size = cv::Size(img.cols/grid_resolution.width, img.rows/grid_resolution.height);

    for ( int i = 0; i < rect_vec.size(); ++i )
    {
        rect = rect_vec[i];
        cv::Point pt1 = cv::Point(rect.x , rect.y);
        cv::Point pt2 = cv::Point(rect.x + rect.width , rect.y + rect.height);
        //std::cout << img.size() << std::endl;
        //std::cout << pt1 << std::endl;
        //std::cout << pt2 << std::endl;
        double intensity = (autocov_deriv.at<double>(0, i)) * 255;
        intensity = std::min<double>(intensity, 255);
        //std::cout << intensity << std::endl;
        cv::rectangle( canvas, pt1, pt2, intensity, -1);
    }

    double min, max;
    cv::minMaxLoc(canvas, &min, &max);
    canvas = canvas - min;
    canvas = 255 * canvas/(max-min);

    //std::cout << canvas << std::endl;

    cv::Mat img_bgr;
    canvas.convertTo(canvas, CV_8UC3);
    //std::cout << canvas << std::endl;
    cv::applyColorMap(canvas, canvas, cv::COLORMAP_JET);
    cv::cvtColor(img, img_bgr, cv::COLOR_GRAY2BGR);
    
    cv::Mat out;
    cv::addWeighted(img_bgr, 0.5, canvas, 0.5, 0, out);
    cv::imshow("Heatmap", out);
    cv::waitKey(0);
}

/*
            heat_map.convertTo(heat_map, CV_8UC1);
            cv::applyColorMap(heat_map, heatmap_img, cv::COLORMAP_JET);
            cv::addWeighted(heatmap_img, 0.5, heat_map_img_sec, 0.5, 0, heat_map_img_sec);
*/