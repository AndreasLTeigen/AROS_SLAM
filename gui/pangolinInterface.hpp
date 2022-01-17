#ifndef pangolinInterface_hpp
#define pangolinInterface_hpp

#include <shared_mutex>
#include <pangolin/pangolin.h>

#include "yaml-cpp/yaml.h"


struct viewPoint{
    double x, y, z;
};

class GUI
{
    private:
        bool shut_down = false;
        int size_x, size_y;
        int menu_bar_width;
        double camera_size, camera_line_width, point_size, line_width;

        mutable std::shared_mutex mutex_shutdown;

    public:
        GUI();
        ~GUI();

        bool getShutdown();

        void setShutdown(bool value);
        void GUIConfigParser(YAML::Node &config);


        void updateFrame(std::shared_ptr<FTracker> tracker);
        void drawEgoMotionLines( std::shared_ptr<FTracker> tracker );
        void drawEgoMotionPoints( std::shared_ptr<FTracker> tracker );
        void drawMapPoints( std::shared_ptr<FTracker> tracker );
        void drawMapPointsOfCurrentFrame( std::shared_ptr<FTracker> tracker );
        void drawCamera(pangolin::OpenGlMatrix &T_wc);
};
int example_fun();

#endif