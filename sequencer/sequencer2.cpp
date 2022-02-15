#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include "sequencer2.hpp"


using std::vector;
using std::string;

Sequencer2::Sequencer2(string source, int playback_mode, int start_idx)
{
    this->curr_idx = start_idx;
    this->playback_mode = playback_mode;

    if (source == "webcam")
    {
        this->webcam = webcam;
        cv::namedWindow("Webcam", cv::WINDOW_AUTOSIZE);
    }
    else
    {
        this->video_path = source;
        cv::namedWindow(this->video_path, cv::WINDOW_AUTOSIZE);
    }


    if ( this->webcam )
    {
        this->video_cap = cv::VideoCapture(0);
    }
    else
    {
        this->video_cap = cv::VideoCapture(this->video_path);
    }

    
    if ( !this->video_cap.isOpened() )
    {
        std::cerr << "ERROR: Could not open video source: " << source << std::endl;
    }
}

Sequencer2::~Sequencer2()
{
    //TODO: Implement sequencer destructor
}

int Sequencer2::getCurrentIdx()
{
    return this->curr_idx;
}

string Sequencer2::getCurrentImgName()
{
    return std::to_string(this->curr_idx);
}

cv::Mat Sequencer2::getCurrentImg()
{
    cv::Mat img;

    if ( this->webcam )
    {
        this->video_cap >> img;
    }
    return img;
}

void Sequencer2::visualizeImage(cv::Mat& img)
{
    cv::imshow("Webcam", img);
}

bool Sequencer2::hasNextFrame()
{
    return true;
}

void Sequencer2::iterateToNewFrame()
{
    this->curr_idx += 1;
}

bool Sequencer2::isFinished()
{
    return false;
}