#include <iostream>
#include <shared_mutex>
#include <opencv2/opencv.hpp>

#include "yaml-cpp/yaml.h"

#include "tracking/tracking.hpp"
#include "sequencer/sequencer3.hpp"
#include "util/util.hpp"

#include "AVGSlam.hpp"

#include "test/helloPangolin.hpp"
#include "test/concurrencyTest.hpp"

using std::string;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::high_resolution_clock;


AVGSlam::AVGSlam( YAML::Node sys_config, YAML::Node data_config, std::shared_ptr<Sequencer3> seq, std::string out_path )
{
    // =====================================================
    //              Parameter Initialization 
    // =====================================================
    
    std::cout << "Initializing SLAM pipeline..." << std::endl;

    this->is_shutdown = false;
    this->visualize = sys_config["UI.display_img"].as<bool>();

    if (sys_config["UI.GUI_show"].as<bool>())
    {
        this->is_main_thread = false;
    }
    else
    {
        this->is_main_thread = true;
    }

    // Initializing camera parameters
    double fx = data_config["Camera.fx"].as<double>();
    double fy = data_config["Camera.fy"].as<double>();
    double cx = data_config["Camera.cx"].as<double>();
    double cy = data_config["Camera.cy"].as<double>();

    // Tracking init parameters
    int Init_frame_buffer_size = sys_config["Trck.init_frame_buffer_size"].as<int>();

    // Initializing other parameters
    int idx_current;
    std::string name_current;
    cv::Mat img_current, img_disp, T_0; 

    this->K_matrix = compileKMatrix( fx, fy, cx, cy );
    


    // ==================================================================
    //          Sequencer and frame tracker Initialization 
    // ==================================================================

    // Initialize this->sequencer and frame tracker
    this->tracker = std::make_shared<FTracker>(sys_config, seq->getSequenceName());
    this->seq = seq;

    if (out_path != "")
    {
        this->tracker->setOutPath(out_path);
    }

    if (sys_config["Trck.out.clear_at_init"].as<bool>())
    {
        clearFile(this->tracker->getOutPath());
    }

    for ( int i = 0; i < Init_frame_buffer_size; i++)
    {
        //img_current = this->seq->getCurrentFrame();
        img_current = this->seq->getCurrentImg();
        idx_current = this->seq->getCurrentIndex();
        name_current = this->seq->getCurrentName();
        if (this->tracker->isShowLog())
        {
            std::cout << "Preloading img: " << name_current << std::endl;
        }
        this->tracker->initializeTracking(img_current, idx_current, this->K_matrix);
        T_0 = cv::Mat::eye(4,4,CV_64F);
        if ( this->tracker->isOutSave() )
        {
            writeTransformation2File(this->tracker->getOutPath(), name_current, T_0 );
        }
        this->seq->iterateToNewFrame();
    }

    std::cout << "SLAM initialization finished..." << std::endl;
}

int AVGSlam::run()
{
    // =====================================================
    //                  Image loop 
    // =====================================================

    int trck_success, idx_current;
    std::string name_current;
    cv::Mat img_current, img_disp, T_global;

    std::cout << "Running SLAM..." << std::endl;
    // Start reading image this->sequence
	while ( this->seq->hasNextImg() && !this->getShutdown() ){
        auto frame_start_time = high_resolution_clock::now();

        // Loading this->sequencer image
        img_current = this->seq->getCurrentImg();
        idx_current = this->seq->getCurrentIndex();
        if (this->tracker->isShowLog())
        {
            std::cout << "Img name: " << this->seq->getCurrentName() << std::endl;
            std::cout << "Img nr: " << this->seq->getCurrentIndex() << std::endl;
        }


        // Computing based on image
        auto computing_start_time = high_resolution_clock::now();
        trck_success = this->tracker->trackFrame(img_current, idx_current, this->K_matrix);
        if (trck_success == 1)
        {
            std::cout << "Fatal error: Quitting program..." << std::endl;
            seq->setFinishedFlag(true);
            break;
        }
        auto computing_end_time = high_resolution_clock::now();

        if (trck_success == 0)
        {
            if ( this->tracker->isOutSave() )
            {
                //Saving frame ego-motion transformation matrix to file
                name_current = this->seq->getCurrentName();
                T_global = this->tracker->getGlobalPose();
                writeTransformation2File(this->tracker->getOutPath(), name_current, T_global );
            }
            
            if ( this->isVisualize() )
            {
                // Visualizing sequencer frame
                //cv::cvtColor(img_current, img_disp, cv::COLOR_GRAY2BGR);
                this->tracker->drawKeypoints(img_current, img_disp);
                //reduceImgContrast(img_disp);
                //this->tracker->drawEpipolarLinesWithPrev(img_disp);
                this->tracker->drawEpipoleWithPrev(img_disp);
                this->tracker->drawKeypointTrails(img_disp);
                this->seq->setFinishedImg(img_disp);
                if ( this->isMainThread())
                {
                    this->seq->visualizeImg(img_disp);
                }
            }

            if (this->tracker->doAnalysis())
            {
                this->tracker->analysis();
            }
        }

		// Set sequencer variables for next iteration
        this->seq->iterateToNewFrame();
		if ( this->seq->isFinished() ){
            break;
		}

        // Timer calculation
        auto frame_end_time = high_resolution_clock::now();
        if ( this->tracker->isShowTimings() )
        {
            auto frame_loading_duration = duration_cast<milliseconds>(computing_start_time-frame_start_time);
            auto frame_visualization_duration = duration_cast<milliseconds>(frame_end_time-computing_end_time);
            auto frame_start_to_stop_duration = duration_cast<milliseconds>(frame_end_time-frame_start_time);
            std::cout << "Frame load time: " << frame_loading_duration.count() << "ms" << std::endl;
            std::cout << "Frame visualization time: " << frame_visualization_duration.count() << "ms" << std::endl;
            std::cout << "Total frame time: " << frame_start_to_stop_duration.count() << "ms" << std::endl;
            std::cout << "------------------------------------------" << std::endl;
        }
    }

	return 0;
    
}

bool AVGSlam::isVisualize()
{
    return this->visualize;
}

bool AVGSlam::isMainThread()
{
    return this->is_main_thread;
}

bool AVGSlam::getShutdown()
{
    return this->is_shutdown;
}

std::shared_ptr<FTracker> AVGSlam::getTracker()
{
    return this->tracker;
}


void AVGSlam::setShutdown( bool value )
{
    this->is_shutdown = value;
}

/* TODO: when going from frame loader or pausing/playing sequencer
    an image is loaded 2 times, this vil disrupt the real validity
    of visualized matches with previous frame when pausing and 
    playing video*/

/*TODO: Returning a vector returns a copy of that vector, make sure that
    vectors are not returned unnecessarily*/