#ifndef guiutil_h
#define guiutil_h

#include <pangolin/pangolin.h>

pangolin::OpenGlMatrix T2OpenGlCameraMatrixFull(cv::Mat T);

pangolin::OpenGlMatrix T2OpenGlCameraMatrixTrans(cv::Mat T);

#endif