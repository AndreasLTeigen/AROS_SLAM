#include <math.h>
#include <memory>
#include <vector>
#include <iostream>
#include <shared_mutex>
#include <opencv2/opencv.hpp>
#include <pangolin/pangolin.h>

#include "yaml-cpp/yaml.h"

#include "guiUtil.hpp"
#include "../util/util.hpp"
#include "../tracking/tracking.hpp"
#include "pangolinInterface.hpp"

using std::vector;
using std::shared_ptr;


GUI::GUI()
{
    //TODO: Implement constructor
}

GUI::~GUI()
{
    //TODO: Implement destructor
}

void GUI::GUIConfigParser(YAML::Node &config)
{
    this->size_x = config["UI.GUI_size_x"].as<int>();
    this->size_y = config["UI.GUI_size_y"].as<int>();
    this->menu_bar_width = config["UI.menu_bar_width"].as<int>();
    this->camera_size = config["UI.camera_size"].as<double>();
    this->camera_line_width = config["UI.camera_line_width"].as<double>();
    this->point_size = config["UI.point_size"].as<double>();
    this->line_width = config["UI.line_width"].as<double>();
}

void GUI::run(std::shared_ptr<FTracker> tracker)
{
    bool perspective_view_toggle = false; // true -> perspective has just been toggled
    bool top_follow = true; // toggle between top and cam follow
    pangolin::OpenGlMatrix T_wc_openGl, O_w_openGl;
    cv::Mat temp_T_wc_1;
    shared_ptr<FrameData> temp_frame1, temp_frame2;

    const int width = tracker->getTrackingFrames()[0]->getImg().cols;
    const int height = tracker->getTrackingFrames()[0]->getImg().rows;
    

    pangolin::CreateWindowAndBind("Main",this->size_x,this->size_y);
    //pangolin::CreateWindowAndBind("Main",this->size_x,2*this->size_y);

    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    // Issue specific OpenGl we might need (
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);


    // Set up a menu bar
    pangolin::CreatePanel("menu").SetBounds(0.0,1.0,0.0,pangolin::Attach::Pix(this->menu_bar_width));
    pangolin::Var<bool> menuFollowCamera("menu.Follow Camera",true,true);
    pangolin::Var<bool> menuTopView("menu.Top View",false,false);
    pangolin::Var<bool> menuCameraView("menu.Camera View",false,false);


    // ==================================== 
    //  Initializing point cloud parameters 
    // ==================================== 
    // Define Projection and initial ModelView matrix
    pangolin::OpenGlRenderState pc_cam(
        pangolin::ProjectionMatrix(this->size_x,this->size_y/2.0,2000,2000,512,389,0.1,1000),
        pangolin::ModelViewLookAt(0,-75, -0.1, 0,0,0,0.0,-1.0, 0.0) // Top camera view
    );
    

    // Create Interactive View in window
    pangolin::View& d_cam = pangolin::Display("point_cloud")
        .SetAspect(-float(width)/float(height))
        //.SetBounds(0.0, 0.5, 0.0, 1.0, 2*float(this->size_x)/float(this->size_y))
        .SetHandler(new pangolin::Handler3D(pc_cam));

    //pangolin::View& d_cam = pangolin::Display("point_cloud")
    //    .SetBounds(0.0, 0.5, pangolin::Attach::Pix(this->menu_bar_width), 1.0, -1024.0f/768.0f)
    //    .SetHandler(new pangolin::Handler3D(pc_cam));


    // ==================================== 
    //  Initializing image view parameters 
    // ==================================== 

    pangolin::GlTexture imageTexture(width,height,GL_RGB,false,0,GL_RGB,GL_UNSIGNED_BYTE);
    pangolin::View& d_img = pangolin::Display("video_feed")
        .SetAspect(float(width)/float(height));
        //.SetBounds(0.5, 1.0, 0.0, 1.0, -float(width)/float(height));




    pangolin::Display("multi")
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(this->menu_bar_width), 1.0)
        .SetLayout(pangolin::LayoutEqual)
        .AddDisplay(d_img)
        .AddDisplay(d_cam);


    int current_frame_nr = tracker->getFrame(-1)->getFrameNr();
    cv::Mat img;
    unsigned char* imageArray = new unsigned char[3*width*height];

    while( !pangolin::ShouldQuit() )
    {
        // Clear screen and activate view to render into
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        
        temp_frame1 = tracker->getFrame(-1);
        temp_T_wc_1 = temp_frame1->getGlobalPose();

        if ( menuFollowCamera && menuTopView )
        {
            menuTopView = false;
            top_follow = true;
            pc_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(0,-50, -0.1, 0,0,0,0.0,-1.0, 0.0)); // Top camera view
        }

        else if ( menuFollowCamera && menuCameraView )
        {
            menuCameraView = false;
            top_follow = false;
            pc_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(0,-1, -10, 0,0,0,0.0,0.0, 1.0)); // Behind camera view
        }

        T_wc_openGl = T2OpenGlCameraMatrixFull(temp_T_wc_1);

        if( menuFollowCamera )
        {
            if ( top_follow )
            {
                O_w_openGl = T2OpenGlCameraMatrixTrans(temp_T_wc_1);
                pc_cam.Follow(O_w_openGl);
            }
            else
            {
                pc_cam.Follow(T_wc_openGl);
            }
        }

        d_cam.Activate(pc_cam);

        // Drawing camera shape
        drawCamera(T_wc_openGl);


        // Drawing ego-motion lines
        this->drawEgoMotionLines( tracker );

        // Drawing ego-motion points
        this->drawEgoMotionPoints( tracker );

        // Drawing mapPoints
        this->drawMapPoints( tracker );

        // Drawing only mapPoints seen in the last frame
        //this->drawMapPointsOfCurrentFrame( tracker );



        // Drawing video feed
        d_img.Activate();
        if (current_frame_nr != temp_frame1->getFrameNr() && !temp_frame1->getImg().empty())
        {
            imageArray = temp_frame1->getImg().data;
            current_frame_nr = temp_frame1->getFrameNr();
        }
        //setImageData(imageArray,3*width*height);
        imageTexture.Upload(imageArray,GL_RGB,GL_UNSIGNED_BYTE);
        glColor4f(1.0f,1.0f,1.0f,1.0f);
        imageTexture.RenderToViewport();
        
        // Swap frames and Process Events
        pangolin::FinishFrame();
    }
}

void GUI::drawEgoMotionLines( std::shared_ptr<FTracker> tracker )
{
    cv::Mat temp_T_wc_1, temp_T_wc_2, temp_t_wc_1, temp_t_wc_2;
    shared_ptr<FrameData> temp_frame1, temp_frame2;

    glLineWidth(this->line_width);
    glColor4f(0.0f,1.0f,0.0f,0.6f);
    glBegin(GL_LINES);

    for( int i = 1; i < tracker->getFrameListLength(); i++ )
    {
        temp_frame1 = tracker->getFrame(i-1);
        temp_frame2 = tracker->getFrame(i);

        temp_T_wc_1 = temp_frame1->getGlobalPose();
        temp_t_wc_1 = T2Trans(temp_T_wc_1);

        temp_T_wc_2 = temp_frame2->getGlobalPose();
        temp_t_wc_2 = T2Trans(temp_T_wc_2);

        glVertex3f(temp_t_wc_1.at<double>(0,0), temp_t_wc_1.at<double>(1,0), temp_t_wc_1.at<double>(2,0));
        glVertex3f(temp_t_wc_2.at<double>(0,0), temp_t_wc_2.at<double>(1,0), temp_t_wc_2.at<double>(2,0));
    }
    glEnd();
}

void GUI::drawEgoMotionPoints( std::shared_ptr<FTracker> tracker )
{
    // Drawing ego-motion points
    cv::Mat temp_T_wc, temp_t_wc;
    shared_ptr<FrameData> temp_frame;
    temp_T_wc = cv::Mat::zeros(4,4,CV_64F);
    temp_t_wc = cv::Mat::zeros(3,1,CV_64F);

    glPointSize(this->point_size);
    glBegin(GL_POINTS);
    glColor3f(1.0,1.0,1.0);

    for( int i = 0; i < tracker->getFrameListLength(); i++ )
    {
        temp_frame = tracker->getFrame(i);
        temp_T_wc = temp_frame->getGlobalPose();
        temp_t_wc = T2Trans(temp_T_wc);

        glVertex3f( temp_t_wc.at<double>(0,0), temp_t_wc.at<double>(1,0), temp_t_wc.at<double>(2,0) );
    }
    glEnd();
}

void GUI::drawMapPoints( std::shared_ptr<FTracker> tracker )
{
    std::shared_ptr<MapPoint> map_point;
    std::shared_ptr<Map3D> map_3d = tracker->getMap3D();

    glPointSize(this->point_size);
    glBegin(GL_POINTS);
    glColor3f(1.0f,0.0f,0.0f);

    for ( int i = 0; i < map_3d->getNumMapPoints(); i++ )
    {
        map_point = map_3d->getMapPoint(i);
        //std::cout << "Map point loc:" << "\nX:" << map_point->getCoordX() << "\nY:" << map_point->getCoordY() << "\nZ:" << map_point->getCoordZ()  << std::endl;
        glVertex3f( map_point->getCoordX(), map_point->getCoordY(), map_point->getCoordZ() );
    }
    glEnd();
}

void GUI::drawMapPointsOfCurrentFrame( std::shared_ptr<FTracker> tracker )
{
    std::shared_ptr<MapPoint> map_point;
    std::shared_ptr<FrameData> curr_frame = tracker->getFrame(-1);
    std::vector<std::shared_ptr<KeyPoint2>> keypoints = curr_frame->getMatchedKeypoints( curr_frame->getFrameNr() - 1 );

    glPointSize(this->point_size);
    glBegin(GL_POINTS);
    glColor3f(1.0f,0.0f,0.0f);

    for ( int i = 0; i < keypoints.size(); i++ )
    {
        if (keypoints[i]->getMapPoint() != nullptr)
        {
            map_point = keypoints[i]->getMapPoint();
            glVertex3f( map_point->getCoordX(), map_point->getCoordY(), map_point->getCoordZ() );
        }
    }
    glEnd();
}

void GUI::drawCamera(pangolin::OpenGlMatrix &T_wc)
{
    double w = this->camera_size;
    double h = this->camera_size*0.75;
    double z = this->camera_size*0.6;

    glPushMatrix();
    glMultMatrixd(T_wc.m);

    glLineWidth(this->camera_line_width);
    glColor4f(0.0f,0.0f,1.0f,0.6f);
    glBegin(GL_LINES);

    glVertex3f(0,0,0);
    glVertex3f(w,h,z);
    glVertex3f(0,0,0);
    glVertex3f(w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,h,z);

    glVertex3f(w,h,z);
    glVertex3f(w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(-w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(w,h,z);

    glVertex3f(-w,-h,z);
    glVertex3f(w,-h,z);
    glEnd();

    glPopMatrix();
}

void GUI::setImageData(unsigned char * imageArray, int size){
  for(int i = 0 ; i < size;i++) {
    imageArray[i] = (unsigned char)(rand()/(RAND_MAX/255.0));
  }
}

int example_fun()
{
    pangolin::CreateWindowAndBind("Main",640,480);
    glEnable(GL_DEPTH_TEST);

    // PointCloud: Define Projection and initial ModelView matrix
    pangolin::OpenGlRenderState pc_cam(
        pangolin::ProjectionMatrix(640,480,420,420,320,240,0.2,100),
        pangolin::ModelViewLookAt(-2,2,-2, 0,0,0, pangolin::AxisY)
    );

    // PointCloud: Create Interactive View in window
    pangolin::Handler3D handler(pc_cam);
    pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f/480.0f)
            .SetHandler(&handler);

    while( !pangolin::ShouldQuit() )
    {
        // Clear screen and activate view to render into
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(pc_cam);

        // Render OpenGL Cube
        pangolin::glDrawColouredCube();

        // Swap frames and Process Events
        pangolin::FinishFrame();
    }
    
    return 0;
}