#ifndef AVGSlam_h
#define AVGSlam_h

#include "tracking/tracking.hpp"
#include "sequencer/sequencer2.hpp"

class AVGSlam
{
    private:
        bool visualize, is_shutdown, is_main_thread;

        std::shared_ptr<FTracker> tracker;
        std::shared_ptr<Sequencer2> seq;
        cv::Mat K_matrix;

    public:
        AVGSlam( YAML::Node sys_config, YAML::Node data_config, std::shared_ptr<Sequencer2> seq, std::string out_path = "");
        ~AVGSlam(){};

        int run();

        bool isVisualize();
        bool isMainThread();
        bool getShutdown();
        std::shared_ptr<FTracker> getTracker();

        void setShutdown(bool value);
};

#endif