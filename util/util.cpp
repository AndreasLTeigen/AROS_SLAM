#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "util.hpp"
#include "yaml-cpp/yaml.h"

using std::string;

void reduceImgContrast(cv::Mat img, int lower_level, int upper_level)
{
    /* Function for moving the contrast of the image to the newly defined
       range <upper_bound - lower_bound> 
       Note: This function results in throwing the [WARN:0] at runtime */
    img = ( img * lower_level / upper_level ) + lower_level;
}

void drawCircle(cv::Mat &img, cv::Point point, int radius)
{
    cv::Scalar color_red = cv::Scalar( 0, 0, 255, 128 );
    std::cout << "Drawing circle at: " << point << std::endl;
    cv::circle(img, point, radius, color_red, 2);
}

void drawCircle(cv::Mat &img, cv::Mat point_mat, int radius)
{
    cv::Point point = cv::Point(point_mat.at<double>(0), point_mat.at<double>(1));
    drawCircle(img, point, radius);
}

cv::Mat fundamentalFromEssential(cv::Mat E_matrix, cv::Mat K_matrix)
{
    cv::Mat K_inv = K_matrix.inv();
    cv::Mat F = K_inv.t()*E_matrix*K_inv;
    return F;
}

cv::Mat calculateEpipole(cv::Mat E_matrix)
{
    /* Caclulates the epipole of in frame 2 the essential matrix of the
       transformation between frame 1 and frame 2, F_matrix can also
       be used as input */

    cv::Mat epipole;
    Eigen::Matrix<double, 3, 3> E_matrix_eigen;
    cv::cv2eigen(E_matrix.t(), E_matrix_eigen);
    Eigen::FullPivLU<Eigen::Matrix<double, 3, 3>> lu(E_matrix_eigen);
    Eigen::Matrix<double, 3, 1> epipole_eigen = lu.kernel();
    cv::eigen2cv(epipole_eigen, epipole);
    epipole = epipole/epipole.at<double>(2);
    return epipole;
}

void drawEpipolarLines(cv::Mat F, cv::Mat &img_disp2,
                        std::vector<cv::Point> points1,
                        std::vector<cv::Point> points2)
{
    std::vector<cv::Vec<double,3>> epilines1;
    cv::computeCorrespondEpilines(points1, 1, F, epilines1); //Index starts with 1

    cv::RNG rng(0);
    for(size_t i=0; i<points1.size(); i++)
    {
        /*
        * Epipolar lines of the 1st point set are drawn in the 2nd image and vice-versa
        */
        cv::Scalar color(rng.uniform(0,256),rng.uniform(0,256),rng.uniform(0,256));
        cv::line(img_disp2,
            cv::Point(0,-epilines1[i][2]/epilines1[i][1]),
            cv::Point(img_disp2.cols,-(epilines1[i][2]+epilines1[i][0]*img_disp2.cols)/epilines1[i][1]),
            color);
    }
}

cv::Mat compileKMatrix( double fx, double fy, double cx, double cy )
{
    // Compiles the rotation matrix from the parameters <fx, fy, cx, cy>
    cv::Mat K;
    K = (cv::Mat_<double>(3,3)<<fx,  0,  cx,
                                0,  fy,  cy,
                                0,  0,    1);
    return K;
}

bool isRotationMatrix(cv::Mat &R)
{
    //Checks if <R> is a valid rotation matrix
    cv::Mat Rt;
    transpose(R, Rt);
    cv::Mat shouldBeIdentity = Rt * R;
    cv::Mat I = cv::Mat::eye(3,3, shouldBeIdentity.type());
    return  cv::norm(I, shouldBeIdentity) < 1e-6;
}

std::vector<double> rotationMatrixToEulerAngles(cv::Mat &R)
{
    assert(isRotationMatrix(R));

    double sy = sqrt(R.at<double>(0,0) * R.at<double>(0,0) +  R.at<double>(1,0) * R.at<double>(1,0) );

    bool singular = sy < 1e-6; // If

    double rx, ry, rz;
    if (!singular)
    {
        rx = atan2(R.at<double>(2,1) , R.at<double>(2,2));
        ry = atan2(-R.at<double>(2,0), sy);
        rz = atan2(R.at<double>(1,0), R.at<double>(0,0));
    }
    else
    {
        rx = atan2(-R.at<double>(1,2), R.at<double>(1,1));
        ry = atan2(-R.at<double>(2,0), sy);
        rz = 0;
    }
    return std::vector<double> {rx, ry, rz};
}

cv::Mat T2Rot(cv::Mat &T)
{
    // Check if getting rows and cols is quicker than this process
    cv::Mat rot = cv::Mat::zeros(3,3,CV_64F);
    rot = (cv::Mat_<double>(3,3)<<T.at<double>(0,0), T.at<double>(0,1), T.at<double>(0,2),
                                T.at<double>(1,0), T.at<double>(1,1), T.at<double>(1,2),
                                T.at<double>(2,0), T.at<double>(2,1), T.at<double>(2,2));
    return rot;
}

cv::Mat T2Trans(cv::Mat &T)
{
    // Chekc if getting rows and cols is quicker than this process
    cv::Mat trans = cv::Mat::zeros(3,1,T.type());
    trans = (cv::Mat_<double>(3,1)<<T.at<double>(0,3),
                                T.at<double>(1,3),
                                T.at<double>(2,3));
    return trans;
}

void T2RotAndTrans(cv::Mat &T, cv::Mat &R, cv::Mat &t)
{
    /*@TODO: Add an assertion to the dimensions of R and t*/

    R = T2Rot(T);

    t = T2Trans(T);
}

cv::Mat inverseTMatrix(cv::Mat T)
{
    cv::Mat T_inv = cv::Mat::eye(4,4,CV_64F);
    cv::Mat R(3,3,CV_64F);
    cv::Mat t(3,1,CV_64F);
    T2RotAndTrans(T, R, t);
    T_inv.rowRange(0,3).colRange(0,3) = R.t();
    T_inv.rowRange(0,3).col(3) = -R.t() * t;
    return T_inv;
}

void dehomogenizeMatrix(cv::Mat& X)
{
    int num_rows = X.rows;

    #pragma omp parallel for
    for ( int i = 0; i < X.cols; i++ )
    {
        X.col(i) = X.col(i) / X.at<double>(num_rows-1,i);
    }
}

std::vector<double> transform2stdParam(cv::Mat &T)
{
    cv::Mat R, t;
    R = cv::Mat::zeros(3,3,CV_64F);
    t = cv::Mat::zeros(3,1,CV_64F);
    T2RotAndTrans(T, R, t);
    std::vector<double> rot = rotationMatrixToEulerAngles(R);
    return std::vector<double> {rot[0], rot[1], rot[2], t.at<double>(0,0), t.at<double>(1,0), t.at<double>(2,0)};
}

void writeParameters2File(std::string file_path, std::string image_idenifier, cv::Mat &T )
{
    std::vector<double> p = transform2stdParam(T);

    std::ofstream pose_file;
    pose_file.open(file_path, std::ios_base::app);
    if (pose_file.is_open())
    {
        pose_file << image_idenifier << " " << p[0] << " " << p[1] << " " << p[2]
                                    << " " << p[3] << " " << p[4] << " " << p[5];
        pose_file << "\n";
        pose_file.close();
    }
    else
    {
        std::cout << "Unable to open file: " << file_path << std::endl;
    }
}