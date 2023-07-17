#ifndef sequencer3_h
#define sequencer3_h

#include <filesystem>
#include <shared_mutex>
#include "opencv2/opencv.hpp"


class Sequencer3
{
    private:
        bool grayscale;
        bool finished = false;
        bool play_mode;
        bool frame_step = false;
        int frame_skip;
        int current_idx;
        int max_idx;
        int fps_target;
        bool timing;
        std::string sequence_name;
        std::string folder_path;
        std::string file_format;
        std::vector<std::string> img_seq;
        std::chrono::time_point<std::chrono::high_resolution_clock> time_prev_frame;

        cv::Mat frame_finished;

        mutable std::shared_mutex mutex_frame_finished;

    public:
        Sequencer3( YAML::Node sys_config, YAML::Node data_config, int seq_nr, bool grayscale );
        ~Sequencer3(){};
        bool loadImgPaths();

        // Read functions
        bool hasNextImg();
        int getCurrentIndex();
        std::string getSequenceName();
        std::string getCurrentPath();
        std::string getCurrentName();
        cv::Mat getCurrentImg();
        void visualizeImg(cv::Mat& img);
        cv::Mat getVisualizationImg();
        void togglePlayMode();
        void toggleFrameStep();
        void addPercentagePrint(cv::Mat& img);
        void addFPSPrint(cv::Mat& img);
        bool isFinished();

        // Write funcitons
        void setFinishedFlag(bool val);
        void setCurrentIndex(int idx);
        void setFinishedImg(cv::Mat& img);
        void iterateToNewFrame();
};


#endif