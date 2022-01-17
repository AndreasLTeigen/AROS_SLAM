#include <thread>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "yaml-cpp/yaml.h"

#include "tracking/tracking.hpp"
#include "sequencer/sequencer.hpp"
#include "util/util.hpp"

#ifdef PANGOLIN_ACTIVE
    #include "gui/pangolinInterface.hpp"
#endif

#include "test/concurrencyTest.hpp"

using std::string;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::high_resolution_clock;


int AVGSlam()
{
    // ###################### Parameter Initialization ######################

    // Load config file
    YAML::Node config = YAML::LoadFile("config/dev_config.yaml");

    // Initializing incoming video data parameters
    const std::string VIn_path = config["VIn.path"].as<std::string>();
    const std::string VIn_file_format = config["VIn.file_format"].as<std::string>();
    int VIn_fps = config["VIn.fps"].as<int>();


    // Initializing camera parameters
    double fx = config["Camera.fx"].as<double>();
    double fy = config["Camera.fy"].as<double>();
    double cx = config["Camera.cx"].as<double>();
    double cy = config["Camera.cy"].as<double>();

    // Initializing method choices
    const std::string detect_method = config["Method.detector"].as<std::string>();
    const std::string desc_method = config["Method.descriptor"].as<std::string>();
    const std::string matching_method = config["Method.matcher"].as<std::string>();
    const std::string pose_calculation_method = config["Method.pose_calculator"].as<std::string>();
    const std::string point_reg_3D_method = config["Method.point_reg_3D"].as<std::string>();
    const std::string point_cull_3D_method = config["Method.point_cull_3D"].as<std::string>();

    // Initializing video output parameters
    bool VOut_show = config["VOut.show"].as<bool>();
    bool VOut_record = config["VOut.record"].as<bool>();
    std::string VOut_rec_name = config["VOut.rec_name"].as<std::string>();
    std::string VOut_rec_path = config["VOut.rec_path"].as<std::string>();
    std::string VOut_full_dst = VOut_rec_path + VOut_rec_name;

    // Initializing output log parameters
    bool Log_save = config["Log.save"].as<bool>();
    std::string Log_name = config["Log.name"].as<std::string>();
    std::string Log_path = config["Log.path"].as<std::string>();
    std::string Log_full_dst = Log_path + Log_name;

    // Initializing user interface parameters
    bool UI_timing_show = config["UI.timing_show"].as<bool>();
    bool UI_tracking_log_show = config["UI.tracking_log_show"].as<bool>();
    bool UI_GUI_show = config["UI.GUI_show"].as<bool>();

    // Initializing sequencer parameters
    int Seq_starting_frame_nr = config["Seq.starting_frame_nr"].as<int>();
    int Seq_frame_buffer_size = config["Seq.frame_buffer_size"].as<int>();

    // Initializing tracker parameters
    int Trck_tracking_window_length = config["Trck.tracking_window_length"].as<int>();
    int Trck_keypoint_trail_length = config["Trck.keypoint_trail_length"].as<int>();

    // Initializing other parameters
    int idx_current;
    std::string name_current;
    cv::Mat  K_matrix, img_previous, img_current, img_disp, T_global, T_0; 
    Detector detector = getDetectionMethod( detect_method );
    Descriptor descriptor = getDescriptionMethod( desc_method );
    Matcher matcher = getMatchingMethod( matching_method );
    PoseCalculator pose_calculator = getRelativePoseCalculationMethod( pose_calculation_method );
    PointReg3D point_reg_3D = get3DPointRegistrationMethod( point_reg_3D_method );
    PointCull3D point_cull_3D = get3DPointCullingMethod( point_cull_3D_method );

    K_matrix = compileKMatrix( fx, fy, cx, cy );
    

    // ###################### Sequencer and frame tracker Initialization ######################

    // Initialize sequencer and frame tracker
    //FTracker tracker = FTracker(detector, descriptor, matcher, Trck_tracking_window_length, UI_timing_show);
    std::shared_ptr<FTracker> tracker = std::make_shared<FTracker>(detector, descriptor, matcher, pose_calculator, point_reg_3D, Trck_tracking_window_length, UI_timing_show, UI_tracking_log_show);
	Sequencer seq = Sequencer(VIn_path, Seq_frame_buffer_size, VIn_file_format, VOut_record, VOut_rec_path, VOut_rec_name, VIn_fps);
	
    seq.set_current_index(Seq_starting_frame_nr);
    for ( int i = 0; i < seq.getFrameBufferSize(); i++)
    {
        img_current = seq.get_current_frame();
        idx_current = seq.get_current_index();
        name_current = seq.get_current_name();
        seq.pre_load_frame_buffer_frame(img_current);
        tracker->initializeTracking(img_current, idx_current, K_matrix);
        T_0 = cv::Mat::eye(4,4,CV_64F);
        if ( Log_save )
        {
            writeParameters2File(Log_full_dst, name_current, T_0 );
        }
    }

    // ###################### GUI Intialization and thread start ######################
    #ifdef PANGOLIN_ACTIVE
        std::thread GUI_thread;
        std::shared_ptr<GUI> viewer = std::make_shared<GUI>();
        if (UI_GUI_show)
        {
            viewer->GUIConfigParser(config);
            GUI_thread = std::thread(&GUI::updateFrame, viewer, std::ref(tracker));
        }
    #endif


    // ###################### Image loop ######################

    // Start reading image sequence
	while ( seq.has_next_frame() ){
        auto frame_start_time = high_resolution_clock::now();

        // Loading sequencer image
		img_current = seq.get_current_frame();
        idx_current = seq.get_current_index();
        seq.frame_buffer_push(img_current);
        std::cout << "Img nr: " << seq.get_current_index() << std::endl;


        // Computing based on image
        auto computing_start_time = high_resolution_clock::now();
        tracker->trackFrame(img_current, idx_current, K_matrix);
        auto computing_end_time = high_resolution_clock::now();


        if ( VOut_show )
        {
            // Visualizing sequencer frame
            tracker->drawKeypoints(img_current, img_disp);
            reduceImgContrast(img_disp);
            //tracker->drawEpipolarLinesWithPrev(img_disp);
            tracker->drawKeypointTrails(img_disp, Trck_keypoint_trail_length);
            //tracker->drawEpipoleWithPrev(img_disp);
            seq.visualize_image(img_disp);
        }


        if ( Log_save )
        {
            //Saving frame ego-motion parameters to file
            name_current = seq.get_current_name();
            T_global = tracker->getGlobalPose();
            writeParameters2File(Log_full_dst, name_current, T_global );
        }


		// Set sequencer variables for next iteration
        seq.iterate_to_new_frame();
		if ( seq.is_finished() ){
			cv::waitKey(0);
		}

        // Timer calculation
        auto frame_end_time = high_resolution_clock::now();
        if ( UI_timing_show )
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

    #ifdef PANGOLIN_ACTIVE
        if (UI_GUI_show)
        {
            // Shut down GUI thread
            viewer->setShutdown(true);
            GUI_thread.join();
        }
    #endif

	return 0;
    
}


int main()
{
    //AVGSlam();
    //example_fun();
    //updateFrame();

    std::thread main_thread(AVGSlam);

    main_thread.join();
    return 0;
}

/* TODO: when going from frame loader or pausing/playing sequencer
    an image is loaded 2 times, this vil disrupt the real validity
    of visualized matches with previous frame when pausing and 
    playing video*/

/*TODO: The configuration intitialization is still very long. It is
    kept like this to avoid the YAML dependency in the dataframe config
    file. Can possibly be loaded into tracking and sequencer files
    - Update: no native .as<std::map<>> function for yaml, have to DIY*/

/*TODO: Returning a vector returns a copy of that vector, make sure that
    vectors are not returned unnecessarily*/