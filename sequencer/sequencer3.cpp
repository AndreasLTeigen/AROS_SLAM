#include <iostream>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>

#include "sequencer3.hpp"

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::high_resolution_clock;

Sequencer3::Sequencer3( YAML::Node sys_config, YAML::Node data_config, int seq_nr, bool grayscale )
{
    this->grayscale = grayscale;

    const YAML::Node& all_seq_names = data_config["Data.sequences"];
    std::string sequence_name = all_seq_names[seq_nr].as<std::string>();
    std::cout << "Loading sequence: "  << sequence_name << std::endl;

    this->sequence_name = sequence_name;
    this->folder_path = data_config["Data.folder"].as<std::string>() + sequence_name;
    this->file_format = data_config["Data.file_format"].as<std::string>();
    this->fps_target = data_config["Data.fps"].as<int>();
    this->play_mode = sys_config["Seq.auto_start"].as<bool>();
    this->frame_skip = sys_config["Seq.frame_skip"].as<int>();
    this->current_idx = sys_config["Seq.starting_frame_nr"].as<int>();
    this->timing = false;
    this->loadImgPaths();
}


bool Sequencer3::loadImgPaths()
{
    for (const auto & entry : std::filesystem::directory_iterator(this->folder_path))
    {
        this->img_seq.push_back(entry.path());
    }

    std::sort(this->img_seq.begin(), this->img_seq.end());
    this->max_idx = this->img_seq.size() - 1;

    if ( this->max_idx < 0 )
    {
        std::cerr << "ERROR: No images found in folder: \n" << this->folder_path << std::endl;
        return false;
    }

    if (this->img_seq[0].substr(this->folder_path.length()) == ".DS_Store")
    {
        this->img_seq.erase(img_seq.begin());
    }
    return true;
}

bool Sequencer3::hasNextImg()
{
    return (this->current_idx <= this->max_idx);
}

int Sequencer3::getCurrentIndex()
{
    return this->current_idx;
}

std::string Sequencer3::getSequenceName()
{
    return this->sequence_name;
}

std::string Sequencer3::getCurrentPath()
{
    return this->img_seq[this->current_idx];
}

std::string Sequencer3::getCurrentName()
{
    return this->img_seq[this->current_idx].substr(this->folder_path.length());
}

cv::Mat Sequencer3::getCurrentImg()
{
    std::string img_path = this->img_seq[this->current_idx];

    cv::Mat img_current;
    if ( this->grayscale )
    {
        img_current = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
    }
    else
    {
        img_current = cv::imread(img_path, cv::IMREAD_UNCHANGED);
    }
    return img_current;
}

void Sequencer3::visualizeImg(cv::Mat& img)
{
    /* Legacy function */
}

cv::Mat Sequencer3::getVisualizationImg()
{
    std::shared_lock(this->mutex_frame_finished);
    return this->frame_finished.clone();
}

void Sequencer3::togglePlayMode()
{
    this->play_mode = !this->play_mode;
}

void Sequencer3::toggleFrameStep()
{
    this->frame_step = !this->frame_step;
}

void Sequencer3::addPercentagePrint(cv::Mat& img)
{
    float percent = 100 * float(this->current_idx) / float(this->max_idx);

    std::stringstream stream;
    stream << std::fixed << std::setprecision(2) << percent;
    std::string percent_txt = stream.str() + "%";
    cv::putText(img, percent_txt, cv::Point(img.cols - 120, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
}

void Sequencer3::addFPSPrint(cv::Mat& img)
{
    if (this->timing)
    {
        std::chrono::time_point<std::chrono::high_resolution_clock> now = high_resolution_clock::now();
        auto ms1 = duration_cast<milliseconds>(now - this->time_prev_frame);
        this->time_prev_frame = now;
        float fps = 1000/ms1.count();
        std::stringstream stream;
        stream << std::fixed << std::setprecision(1) << fps;
        std::string fps_txt = stream.str();
        cv::putText(img, fps_txt, cv::Point(img.cols - 120, 60), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        cv::putText(img, "FPS", cv::Point(img.cols - 45, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    }
    else
    {
        this->timing = true;
    }
}

bool Sequencer3::isFinished()
{
    return this->finished;
}

void Sequencer3::setFinishedFlag(bool val)
{
    this->finished = val;
}

void Sequencer3::setCurrentIndex(int idx)
{
    this->current_idx = idx;
}

void Sequencer3::setFinishedImg(cv::Mat& img)
{
    std::unique_lock(this->mutex_frame_finished);
    this->frame_finished = img;
}

void Sequencer3::iterateToNewFrame()
{
    while(!this->play_mode)
    {
        /* 
        Wait loop 
        TODO: Make this into a conditional thread sleep
        */
        if (this->finished)
        {
            break;
        }
        if (this->frame_step)
        {
            this->toggleFrameStep();
            break;
        }
    }

    this->current_idx = std::min(this->max_idx, this->current_idx + this->frame_skip + 1);
    if ( this->current_idx == this->max_idx )
    {
        this->finished = true;
    }
}