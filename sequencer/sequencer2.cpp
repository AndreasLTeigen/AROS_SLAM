#include <iostream>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>

#include "sequencer2.hpp"

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::high_resolution_clock;

Sequencer2::Sequencer2( YAML::Node sys_config, YAML::Node data_config, int seq_nr, bool grayscale )
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


bool Sequencer2::loadImgPaths()
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

bool Sequencer2::hasNextImg()
{
    return (this->current_idx <= this->max_idx);
}
//cv::Mat Sequencer2::getNextImg();

int Sequencer2::getCurrentIndex()
{
    return this->current_idx;
}

std::string Sequencer2::getSequenceName()
{
    return this->sequence_name;
}

std::string Sequencer2::getCurrentPath()
{
    return this->img_seq[this->current_idx];
}

std::string Sequencer2::getCurrentName()
{
    return this->img_seq[this->current_idx].substr(this->folder_path.length());
}

cv::Mat Sequencer2::getCurrentImg()
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

void Sequencer2::visualizeImg(cv::Mat& img)
{
    cv::destroyAllWindows();
    cv::Mat out;
    if (img.channels() == 1){
		cv::cvtColor(img, out, cv::COLOR_GRAY2BGR);
	}
	else{
		out = img;
	}

    this->addPercentagePrint(out);
    this->addFPSPrint(out);
    cv::imshow(this->getCurrentName(), out);

    if (this->play_mode)
    {
        this->keyPress(cv::waitKey(1));
    }
    else
    {
        this->keyPress(cv::waitKey(0));
    }
}

void Sequencer2::keyPress(char key)
{
    key = int(key & 255);
    if (key == 'r')
    {
        this->play_mode = !this->play_mode;
    }
    else if (key == 'q')
    {
        this->play_mode = false;
        this->finished = true;
        cv::destroyAllWindows();
    }
}

void Sequencer2::addPercentagePrint(cv::Mat& img)
{
    float percent = 100 * float(this->current_idx) / float(this->max_idx);

    std::stringstream stream;
    stream << std::fixed << std::setprecision(2) << percent;
    std::string percent_txt = stream.str() + "%";
    cv::putText(img, percent_txt, cv::Point(img.cols - 120, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
}

void Sequencer2::addFPSPrint(cv::Mat& img)
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

bool Sequencer2::isFinished()
{
    return this->finished;
}


void Sequencer2::setCurrentIndex(int idx)
{
    this->current_idx = idx;
}

void Sequencer2::iterateToNewFrame()
{
    this->current_idx = std::min(this->max_idx, this->current_idx + this->frame_skip + 1);
    if ( this->current_idx == this->max_idx )
    {
        this->finished = true;
    }
}