#include <opencv2/opencv.hpp>

#include "findKeypoints.hpp"

using std::vector;

using cv::Mat;
using cv::KeyPoint;
using cv::Ptr;

Detector getDetectionMethod( std::string detect_method )
{
    /* Retrieve the right keypoint detector class from the corresponding config string */

    if ( detect_method == "orb" )
    {
        return Detector::ORB;
    }
    else
    {
        std::cout << "ERROR: DETECTION METHOD NOT FOUND" << std::endl;
        return Detector::NONE;
    }
}

Descriptor getDescriptionMethod( std::string desc_method )
{
    /* Retrieve the right keypoint descriptor class from the corresponding config string */
    if ( desc_method == "orb" )
    {
        return Descriptor::ORB;
    }
    else
    {
        std::cout << "ERROR: DESCRIPTION METHOD NOT FOUND" << std::endl;
        return Descriptor::NONE;
    }
}

void findKeypoints(cv::Mat& img, std::shared_ptr<FrameData> frame, Detector detect_type, Descriptor desc_type )
{
    /* Finds keypoints in <img> and stores them in <frame>,
        NOTE: Currently detector and descriptor either has to both be an opencv method or a custom method */

    bool opencv_method = false;
    vector<cv::KeyPoint> kpts;
    Mat desc;

    switch(detect_type)
    {
        case Detector::ORB:
            Ptr<cv::ORB> detector = cv::ORB::create();
            detector->detect( img, kpts );
            opencv_method = true;
    }

    switch(desc_type)
    {
        case Descriptor::ORB:
            Ptr<cv::ORB> descriptor = cv::ORB::create();
            descriptor->compute( img, kpts, desc);
            opencv_method = true;
    }

    if (opencv_method)
    {
        frame->registerKeypoints(kpts, desc);
    }
}