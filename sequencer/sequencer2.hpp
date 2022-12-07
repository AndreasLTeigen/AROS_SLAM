#ifndef sequencer2_h
#define sequencer2_h

#include <filesystem>
#include "opencv2/opencv.hpp"


class Sequencer2
{
    private:
        bool grayscale;
        bool finished = false;
        bool play_mode;
        int frame_skip;
        int current_idx;
        int max_idx;
        int fps_target;
        bool timing;
        std::string folder_path;
        std::string file_format;
        std::vector<std::string> img_seq;
        std::chrono::time_point<std::chrono::high_resolution_clock> time_prev_frame;

    public:
        Sequencer2( YAML::Node sys_config, YAML::Node data_config, int seq_nr, bool grayscale );
        ~Sequencer2(){};
        bool loadImgPaths();

        // Read functions
        bool hasNextImg();
        int getCurrentIndex();
        std::string getCurrentPath();
        std::string getCurrentName();
        cv::Mat getCurrentImg();
        void visualizeImg(cv::Mat& img);
        void keyPress(char key);
        void addPercentagePrint(cv::Mat& img);
        void addFPSPrint(cv::Mat& img);
        bool isFinished();

        // Write funcitons
        void setCurrentIndex(int idx);
        void iterateToNewFrame();
};


#endif