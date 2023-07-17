#ifndef pangolinInterface_hpp
#define pangolinInterface_hpp

#include <shared_mutex>
#include <pangolin/pangolin.h>

#include "yaml-cpp/yaml.h"


class GUI
{
    private:
        bool shut_down = false;
        bool true_color;
        int map_width, map_height;
        int menu_bar_width;
        double camera_size, camera_line_width, point_size, line_width;

        std::string name_map_gui = "Map";

        mutable std::shared_mutex mutex_shutdown;

    public:
        GUI();
        ~GUI();

        void GUIConfigParser(YAML::Node& sys_config);


        void run(   std::shared_ptr<FTracker> tracker, 
                    std::shared_ptr<Sequencer3> sequencer);
        void run2(  std::shared_ptr<FTracker> tracker, 
                    std::shared_ptr<Sequencer3> sequencer);
        void drawEgoMotionLines( std::shared_ptr<FTracker> tracker );
        void drawEgoMotionPoints( std::shared_ptr<FTracker> tracker );
        void drawMapPoints( std::shared_ptr<FTracker> tracker );
        void drawMapPointsOfCurrentFrame( std::shared_ptr<FTracker> tracker );
        void drawCamera(pangolin::OpenGlMatrix &T_wc);
        void setImageData(unsigned char * imageArray, int size);
};
int example_fun();

#endif