#include <vector>
#include <chrono>
#include <memory>
#include <iostream>
#include <Eigen/Dense>
#include <shared_mutex>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include "tracking.hpp"
#include "../util/util.hpp"
#include "../keypointExtraction/findKeypoints.hpp"
#include "../keypointMatching/matchKeypoints.hpp"
#include "../poseCalculation/poseCalculation.hpp"
#include "../mapPointHandler/mapPointRegistration.hpp"
#include "../sequencer/sequencer.hpp"


using cv::Mat;
using cv::KeyPoint;
using cv::DMatch;
using std::vector;
using std::string;
using std::shared_ptr;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::high_resolution_clock;



FTracker::FTracker(YAML::Node config){
    this->curr_frame_nr = 0;
    this->T_global = cv::Mat::eye(4,4,CV_64F);
    this->detec_type = getDetectionMethod( config["Method.detector"].as<std::string>() );
    this->descr_type = getDescriptionMethod( config["Method.descriptor"].as<std::string>() );
    this->matcher_type = getMatchingMethod( config["Method.matcher"].as<std::string>() );
    this->pose_calculation_type = getRelativePoseCalculationMethod( config["Method.pose_calculator"].as<std::string>() );
    this->point_reg_3D_type = get3DPointRegistrationMethod( config["Method.point_reg_3D"].as<std::string>() );
    this->point_cull_3D_type = get3DPointCullingMethod( config["Method.point_cull_3D"].as<std::string>() );
    this->tracking_window_length = config["Trck.tracking_window_length"].as<int>();
    this->show_timings = config["UI.timing_show"].as<bool>();
    this->show_tracking_log = config["UI.tracking_log_show"].as<bool>();
    this->map_3d = std::make_shared<Map3D>();
}

FTracker::~FTracker()
{
    //@TODO: Create destructor
}

int FTracker::getCurrentFrameNr()
{
    std::shared_lock lock(this->mutex_curr_frame_nr);
    return this->curr_frame_nr;
}

int FTracker::getTrackingWindowLength()
{
    return this->tracking_window_length;
}

int FTracker::getFrameListLength()
{
    std::shared_lock lock(this->mutex_frame_list);
    return this->frame_list.size();
}

Detector FTracker::getDetectorType()
{
    return this->detec_type;
}

Descriptor FTracker::getDescriptorType()
{
    return this->descr_type;
}

Matcher FTracker::getMatcherType()
{
    return this->matcher_type;
}

PoseCalculator FTracker::getPoseCalcuationType()
{
    return this->pose_calculation_type;
}

PointReg3D FTracker::getPointReg3DType()
{
    return this->point_reg_3D_type;
}

Mat FTracker::getGlobalPose()
{
    std::shared_lock lock(this->mutex_T_global);
    return this->T_global;
}

vector<shared_ptr<FrameData>> FTracker::getTrackingFrames()
{
    // <WARNING> <frame_list> will not be protected against race conditions outside the scope of this function 
    std::shared_lock lock(this->mutex_frame_list);
    return this->frame_list;
}

shared_ptr<FrameData> FTracker::getFrame(int index)
{
    /* Returns the <FrameData> pointer in <frame_list> that corresponsd with the index <index>.
       Negative indexes are also valid */
    std::shared_lock lock(this->mutex_frame_list);
    if (index < 0)
    {
        return this->frame_list.end()[index];
    }
    else
    {
        return this->frame_list[index];
    }
}

std::shared_ptr<Map3D> FTracker::getMap3D()
{
    std::shared_lock lock(this->mutex_map3D);
    return this->map_3d;
}

void FTracker::setCurrentFrameNr(int curr_frame_nr)
{
    std::unique_lock lock(this->mutex_curr_frame_nr);
    this->curr_frame_nr = curr_frame_nr;
}

void FTracker::setGlobalPose(cv::Mat T_global)
{
    std::unique_lock lock(this->mutex_T_global);
    this->T_global = T_global;
}

void FTracker::updateGlobalPose(cv::Mat T_rel, shared_ptr<FrameData> current_frame)
{
    this->setGlobalPose(this->getGlobalPose() * T_rel);
    current_frame->setGlobalPose(this->getGlobalPose());
}

void FTracker::initializeTracking(cv::Mat &img, int img_id, Mat K_matrix)
{
    /* Creates a new initalization frame. Currently just extracts the 
       keypoints with descriptor*/

    shared_ptr<FrameData> frame = shared_ptr<FrameData>(new FrameData(this->getCurrentFrameNr(), img_id, K_matrix));

    findKeypoints( img, frame, this->getDetectorType(), this->getDescriptorType());
    this->appendTrackingFrame(frame);
}



void FTracker::trackFrame(cv::Mat &img, int img_id, Mat K_matrix, int comparison_frame_spacing)
{
    /* Core function of FTracker, recieves new image, extracts information
       with chosen methods and redirects to matching / pose prediction
       funcitons */

    shared_ptr<FrameData> frame1 = shared_ptr<FrameData>(new FrameData(this->getCurrentFrameNr(), img_id, K_matrix));
    shared_ptr<FrameData> frame2 = this->getFrame(-comparison_frame_spacing);


    std::cout << "Frame nr: " << frame1->getFrameNr() << std::endl;

    auto kpts_start_time = high_resolution_clock::now();


    // Keypoint identification
    findKeypoints( img, frame1, this->getDetectorType(), this->getDescriptorType());


    auto kpts_end_time = high_resolution_clock::now(); 


    //Keypoint matching
    matchKeypoints( frame1, frame2, this->getMatcherType() );


    auto match_end_time = high_resolution_clock::now();


    // Relative pose calculation
    shared_ptr<Pose> rel_pose = calculateRelativePose(frame1, frame2, K_matrix, this->getPoseCalcuationType());
    
    rel_pose->updateParametrization();
    this->updateGlobalPose(rel_pose->getTMatrix(), frame1);


    auto rel_pose_calc_end_time = high_resolution_clock::now();


    // Map update
    register3DPoints( frame1, frame2, this->getMap3D(), this->getPointReg3DType() );


    auto map_update_end_time = high_resolution_clock::now();


    this->appendTrackingFrame(frame1);

    // Cleanup
    this->frameListPruning();

    auto cleanup_end_time = high_resolution_clock::now();

    if (show_tracking_log)
    {
        std::cout << "Parametrization: \n" << *rel_pose->getParametrization() << std::endl;
        std::cout << "Global Pose: \n" << frame1->getGlobalPose() << std::endl;
    }

    if (show_timings)
    {
        auto ms1 = duration_cast<milliseconds>(kpts_end_time-kpts_start_time);
        auto ms2 = duration_cast<milliseconds>(match_end_time-kpts_end_time);
        auto ms3 = duration_cast<milliseconds>(rel_pose_calc_end_time-match_end_time);
        auto ms4 = duration_cast<milliseconds>(map_update_end_time-rel_pose_calc_end_time);
        auto ms5 = duration_cast<milliseconds>(cleanup_end_time-map_update_end_time);
        std::cout << "Keypoint calculation time: " << ms1.count() << "ms" << std::endl;
        std::cout << "Matching time: " << ms2.count() << "ms" << std::endl;
        std::cout << "Relalive pose calculation time: " << ms3.count() << "ms" << std::endl;
        std::cout << "Map update time: " << ms4.count() << "ms" << std::endl;
        std::cout << "Cleanup time: " << ms5.count() << "ms" << std::endl;
    }
}



void FTracker::appendTrackingFrame(shared_ptr<FrameData> new_frame)
{
    //Adding frame to list of tracking frames
    std::unique_lock lock1(this->mutex_frame_list);
    std::unique_lock lock2(this->mutex_curr_frame_nr);
    this->frame_list.push_back(std::move(new_frame));
    this->curr_frame_nr += 1;
}

void FTracker::frameListPruning()
{
    /* Removing a frame if <frame_list.size()> > <tracking_window_length>
        Speed of vectors .erase function is dependent on lenght left in
        vecor, some other fifo designated container might be better */

    if (this->getFrameListLength() > this->getTrackingWindowLength())
    {
        std::unique_lock lock(this->mutex_frame_list);
        this->frame_list.erase(frame_list.begin());
    }
}

void FTracker::drawKeypoints(cv::Mat &src, cv::Mat &dst, int frame_nr)
{
    /* Draw keypoints from frame number <frame_nr> on the image <img>.
       If <frame_nr> is unspecified draws keyframes from the most recent
       frame 
       $TODO: make this work directly on custom keypoint datastructures */
    
    if (frame_nr == -1)
    {
        frame_nr = this->getFrameListLength()-1;
    }

    std::shared_ptr<FrameData> temp_frame = this->getFrame(frame_nr);
    cv::drawKeypoints(src, temp_frame->compileCVKeypoints(), dst);
}

void FTracker::drawKeypointTrails(cv::Mat &img, int trail_length, int frame_nr, int trail_thickness)
{
    /* Prints a line from the matched keypoints from frame nr <frame_nr> 
       to <frame_nr - max(trail_length, tracking_window_length)>
       Assumption: given more match possibilites optimal forward match 
       is equal to optimal backwards match, and that matches of subsequent
       frames are always available */

    if (frame_nr == -1)
    {
        frame_nr = this->getFrameListLength()-1;
    }

    if (trail_length >= this->tracking_window_length)
    {
        trail_length = this->tracking_window_length - 1;
    }

    vector<shared_ptr<KeyPoint2>> kpts;
    shared_ptr<KeyPoint2> temp_kpt1, temp_kpt2;
    shared_ptr<Match> match;
    cv::Scalar color_blue = cv::Scalar( 255, 0, 0, 128 );

    std::shared_ptr<FrameData> temp_frame = this->getFrame(frame_nr);
    kpts = temp_frame->getKeypoints();

    for (shared_ptr<KeyPoint2> kpt : kpts)
    {
        for (int i = 0; i < trail_length; i++)
        {
            match = kpt->getHighestConfidenceMatch(kpt->getObservationFrameNr()-1);
            if (match != nullptr)
            {
                temp_kpt1 = match->getKpt1();
                temp_kpt2 = match->getKpt2();
                cv::line(img, temp_kpt1->compileCV2DPoint(), temp_kpt2->compileCV2DPoint(), color_blue, trail_thickness);
                kpt = match->getConnectingKpt(kpt->getObservationFrameNr()-1);
            }
            else
            {
                break;
            }
        }
    }
}

void FTracker::drawEpipoleWithPrev(cv::Mat &img_disp, int frame_nr)
{
    cv::Mat E_matrix, F_matrix, epipole;
    std::shared_ptr<FrameData> curr_frame = this->getFrame(frame_nr);

    if (frame_nr == -1)
    {
        frame_nr = this->getFrameListLength()-1;
    }

    E_matrix = curr_frame->getRelPose( curr_frame->getFrameNr()-1 )->getEMatrix();
    F_matrix = fundamentalFromEssential( E_matrix, curr_frame->getKMatrix() );
    epipole = calculateEpipole( F_matrix );
    drawCircle( img_disp, epipole );
}

void FTracker::drawEpipolarLinesWithPrev(cv::Mat &img_disp, int frame_nr)
{
    vector<cv::Point> pts1, pts2;
    cv::Mat E_matrix, F_matrix, epipole;
    std::shared_ptr<FrameData> curr_frame, prev_frame;

    if (frame_nr == -1)
    {
        frame_nr = this->getFrameListLength()-1;
    }
    curr_frame = this->getFrame( frame_nr );
    prev_frame = this->getFrame( frame_nr - 1 );
    pts1 = curr_frame->compileCV2DPoints();
    pts2 = prev_frame->compileCV2DPoints();
    E_matrix = curr_frame->getRelPose( curr_frame->getFrameNr()-1 )->getEMatrix();
    F_matrix = fundamentalFromEssential( E_matrix, curr_frame->getKMatrix() );
    drawEpipolarLines( F_matrix, img_disp, pts2, pts1 );
}

//Functions for error checking

float FTracker::getLongestDistanceMatch(shared_ptr<FrameData> frame1, shared_ptr<FrameData> frame2, shared_ptr<KeyPoint2>& kpt1, shared_ptr<KeyPoint2>& kpt2)
{
    float longest_kpt_dist = 0;
    shared_ptr<Match> temp_match;
    shared_ptr<KeyPoint2> matched_kpt2;
    vector<shared_ptr<KeyPoint2>> matched_kpts1 = frame1->getMatchedKeypoints( frame2->getFrameNr() );
    
    for ( shared_ptr<KeyPoint2> matched_kpt1 : matched_kpts1 )
    {
        temp_match = matched_kpt1->getHighestConfidenceMatch( frame2->getFrameNr() );
        matched_kpt2 = temp_match->getConnectingKpt(frame2->getFrameNr());
        if ( KeyPoint2::calculateKeypointDistance( matched_kpt1, matched_kpt2 ) >= longest_kpt_dist)
        {
            longest_kpt_dist = KeyPoint2::calculateKeypointDistance( matched_kpt1, matched_kpt2 );
            kpt1 = matched_kpt1;
            kpt2 = matched_kpt2;
            //std::cout << matched_kpt1->compileCV2DPoint() << " | " << matched_kpt2->compileCV2DPoint() << std::endl;
        }
    }
    return longest_kpt_dist;
}