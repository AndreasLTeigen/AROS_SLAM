#include <math.h>
#include <memory>
#include <vector>
#include <iostream>
#include <shared_mutex>
#include <opencv2/opencv.hpp>
#include <pangolin/pangolin.h>

#include <opencv2/opencv.hpp>

#include "yaml-cpp/yaml.h"

#include "guiUtil.hpp"
#include "../util/util.hpp"
#include "../tracking/tracking.hpp"
#include "../sequencer/sequencer3.hpp"
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

void GUI::GUIConfigParser(YAML::Node& sys_config)
{
    this->map_width = sys_config["UI.map_width"].as<int>();
    this->map_height = sys_config["UI.map_height"].as<int>();
    this->menu_bar_width = sys_config["UI.menu_bar_width"].as<int>();
    this->camera_size = sys_config["UI.camera_size"].as<double>();
    this->camera_line_width = sys_config["UI.camera_line_width"].as<double>();
    this->point_size = sys_config["UI.point_size"].as<double>();
    this->line_width = sys_config["UI.line_width"].as<double>();
    this->true_color = sys_config["UI.true_color"].as<bool>();
}

void GUI::run(  std::shared_ptr<FTracker> tracker,
                std::shared_ptr<Sequencer3> sequencer )
{
    bool perspective_view_toggle = false; // true -> perspective has just been toggled
    bool top_follow = true; // toggle between top and cam follow
    pangolin::OpenGlMatrix T_wc_openGl, O_w_openGl;
    cv::Mat temp_T_wc_1, temp_T_wc_2, temp_t_wc_1, temp_t_wc_2;
    shared_ptr<FrameData> temp_frame1, temp_frame2;
    temp_T_wc_1 = cv::Mat::zeros(4,4,CV_64F);
    temp_T_wc_2 = cv::Mat::zeros(4,4,CV_64F);
    temp_t_wc_1 = cv::Mat::zeros(3,1,CV_64F);
    temp_t_wc_2 = cv::Mat::zeros(3,1,CV_64F);


    pangolin::CreateWindowAndBind(  this->name_map_gui,
                                    this->map_width,
                                    this->map_height);

    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    // Issue specific OpenGl we might need (
    //TODO: check if these really do anything
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Set up a menu bar
    pangolin::CreatePanel("menu").SetBounds(0.0,1.0,0.0,pangolin::Attach::Pix(this->menu_bar_width));
    pangolin::Var<bool> menuFollowCamera("menu.Follow Camera",true,true);
    pangolin::Var<bool> menuTopView("menu.Top View",false,false);
    pangolin::Var<bool> menuCameraView("menu.Camera View",false,false);

    // Define Projection and initial ModelView matrix
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(this->map_width,this->map_height,2000,2000,512,389,0.1,1000),
        pangolin::ModelViewLookAt(0,-75, -0.1, 0,0,0,0.0,-1.0, 0.0) // Top camera view
    );

    // Create Interactive View in window
    pangolin::View& d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(this->menu_bar_width), 1.0, -1024.0f/768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));

    pangolin::RegisterKeyPressCallback('r', [&](){
        sequencer->togglePlayMode();
        });
    pangolin::RegisterKeyPressCallback('d', [&](){
        sequencer->toggleFrameStep();
        });

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
            s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(0,-50, -0.1, 0,0,0,0.0,-1.0, 0.0)); // Top camera view
        }

        else if ( menuFollowCamera && menuCameraView )
        {
            menuCameraView = false;
            top_follow = false;
            s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(0,-1, -10, 0,0,0,0.0,0.0, 1.0)); // Behind camera view
        }

        T_wc_openGl = T2OpenGlCameraMatrixFull(temp_T_wc_1);

        if( menuFollowCamera )
        {
            if ( top_follow )
            {
                O_w_openGl = T2OpenGlCameraMatrixTrans(temp_T_wc_1);
                s_cam.Follow(O_w_openGl);
            }
            else
            {
                s_cam.Follow(T_wc_openGl);
            }
        }

        d_cam.Activate(s_cam);

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

        // Swap frames and Process Events
        pangolin::FinishFrame();

        cv::Mat img = sequencer->getVisualizationImg();
        if (!img.empty())
        {
            cv::imshow("video feed", img);
        }
        if (sequencer->isFinished())
        {
            break;
        }
    }
    sequencer->setFinishedFlag(true);
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
    float color;
    std::shared_ptr<MapPoint> map_point;
    std::shared_ptr<Map3D> map_3d = tracker->getMap3D();

    glPointSize(this->point_size);
    glBegin(GL_POINTS);
    glColor3f(1.0f,0.0f,0.0f);

    for ( int i = 0; i < map_3d->getNumMapPoints(); i++ )
    {
        map_point = map_3d->getMapPoint(i);
        if (this->true_color)
        {
            color = float(map_point->getColor())/255.0f;
            glColor3f(color,color,color);
        }
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