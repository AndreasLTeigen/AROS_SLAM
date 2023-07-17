#include <opencv2/opencv.hpp>
#include <pangolin/pangolin.h>

#include "guiUtil.hpp"


pangolin::OpenGlMatrix T2OpenGlCameraMatrixFull(cv::Mat T)
{
    cv::Mat R(3,3,CV_64F);
    cv::Mat t(3,1,CV_64F);
    pangolin::OpenGlMatrix T_openGL;

    R = T.rowRange(0,3).colRange(0,3);
    t = T.rowRange(0,3).col(3);

    T_openGL.m[0]   = R.at<double>(0,0);
    T_openGL.m[1]   = R.at<double>(1,0);
    T_openGL.m[2]   = R.at<double>(2,0);
    T_openGL.m[3]   = 0.0;

    T_openGL.m[4]   = R.at<double>(0,1);
    T_openGL.m[5]   = R.at<double>(1,1);
    T_openGL.m[6]   = R.at<double>(2,1);
    T_openGL.m[7]   = 0.0;

    T_openGL.m[8]   = R.at<double>(0,2);
    T_openGL.m[9]   = R.at<double>(1,2);
    T_openGL.m[10]  = R.at<double>(2,2);
    T_openGL.m[11]  = 0.0;

    T_openGL.m[12]  = t.at<double>(0);
    T_openGL.m[13]  = t.at<double>(1);
    T_openGL.m[14]  = t.at<double>(2);
    T_openGL.m[15]  = 1.0;

    return T_openGL;
}

pangolin::OpenGlMatrix T2OpenGlCameraMatrixTrans(cv::Mat T)
{
    cv::Mat t(3,1,CV_64F);
    pangolin::OpenGlMatrix T_openGL;
    T_openGL.SetIdentity();

    t = T.rowRange(0,3).col(3);

    T_openGL.m[12]  = t.at<double>(0);
    T_openGL.m[13]  = t.at<double>(1);
    T_openGL.m[14]  = t.at<double>(2);

    return T_openGL;
}