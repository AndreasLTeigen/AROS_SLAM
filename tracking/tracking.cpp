#include <vector>
#include <chrono>
#include <memory>
#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include <shared_mutex>
#include <yaml-cpp/yaml.h>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include "tracking.hpp"
#include "../util/util.hpp"
#include "../keypointExtraction/keypointExtraction.hpp"
#include "../motionPrior/motionPrior.hpp"
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
using json = nlohmann::json;



FTracker::FTracker(YAML::Node config)
{
    this->curr_frame_nr = 0;
    this->T_global = cv::Mat::eye(4,4,CV_64F);

    this->frame_preprocessor = getPreprocessor( config["Method.preprocessor"].as<std::string>() );
    this->motion_prior = getMotionPrior( config["Method.motion_prior"].as<std::string>() );
    this->extractor = getExtractor( config["Method.extractor"].as<std::string>() );
    this->matcher = getMatcher( config["Method.matcher"].as<std::string>() );
    this->pose_calculator = getPoseCalculator( config["Method.pose_calculator"].as<std::string>() );
    this->map_point_reg = getMapPointRegistrator( config["Method.point_reg_3D"].as<std::string>() );
    this->map_point_cull = getMapPointCuller( config["Method.point_cull_3D"].as<std::string>() );
    this->pose_param = getParametrization( config["Methods.param"].as<std::string>() );

    this->map_3d = std::make_shared<Map3D>();

    this->tracking_window_length = config["Trck.tracking_window_length"].as<int>();
    this->show_analysis = config["Trck.analysis"].as<bool>();
    this->show_timings = config["Trck.timing_show"].as<bool>(); // Split this up or change location
    this->kpt_trail_length = config["Trck.out.kpt_trail_length"].as<int>();
    this->show_log = config["Trck.log.show"].as<bool>();

    this->is_pose_analysis = config["Anlys.pose_calculator"].as<bool>();

    this->out_path = config["Trck.out.path"].as<std::string>() + config["Trck.out.name"].as<std::string>();
    this->save_out = config["Trck.out.save"].as<bool>();
}

FTracker::~FTracker()
{
    //@TODO: Create destructor
}

bool FTracker::isOutSave()
{
    return this->save_out;
}

bool FTracker::isShowLog()
{
    return this->show_log;
}

bool FTracker::isShowTimings()
{
    return this->show_timings;
}

bool FTracker::isShowAnalysis()
{
    return this->show_analysis;
}

std::string FTracker::getOutPath()
{
    return this->out_path;
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

Mat FTracker::getGlobalPose()
{
    std::shared_lock lock(this->mutex_T_global);
    return this->T_global;
}

/*
std::shared_ptr<PoseCalculator> FTracker::getPoseCalculator()
{
    // NOTE: This function is not protected from data races.
    return this->pose_calculator;
}
*/

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

void FTracker::setOutPath(std::string out_path)
{
    this->out_path = out_path;
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

    this->extractor->extract( img, frame, this->getMap3D() );
    frame->setImg( img );
    this->updateGlobalPose(cv::Mat::eye(4,4, CV_64F), frame);
    this->appendTrackingFrame(frame);
}



int FTracker::trackFrame(cv::Mat &img, int img_id, Mat K_matrix, int comparison_frame_spacing)
{
    /* Core function of FTracker, recieves new image, extracts information
       with chosen methods and redirects to matching / pose prediction
       funcitons */

    shared_ptr<FrameData> frame1 = shared_ptr<FrameData>(new FrameData(this->getCurrentFrameNr(), img_id, K_matrix));
    shared_ptr<FrameData> frame2 = this->getFrame(-comparison_frame_spacing);

    frame1->setImg( img ); // TODO: Remove later when not needed anymore


    if (this->isShowLog()) std::cout << "Frame nr: " << frame1->getFrameNr() << std::endl;

    auto preprocess_start_time = high_resolution_clock::now();            // Timer

    // ==================================== 
    //         Frame preprocessing
    // ====================================
    if (this->isShowLog()) std::cout << "Frame preprocessing..." << std::endl;
    this->frame_preprocessor->calculate( img, frame1 );

    auto kpts_start_time = high_resolution_clock::now();            // Timer

    // ==================================== 
    //          Motion prior
    // ====================================
    if (this->isShowLog()) std::cout << "Getting motion prior..." << std::endl;
    this->motion_prior->calculate( frame1, frame2 );


    // ==================================== 
    //      Keypoint identification
    // ==================================== 
    if (this->isShowLog()) std::cout << "Extracting keypoints..." << std::endl;
    this->extractor->extract( img, frame1, this->getMap3D() );


    auto kpts_end_time = high_resolution_clock::now();              // Timer


    // ==================================== 
    //          Keypoint matching
    // ==================================== 
    if (this->isShowLog()) std::cout << "Matching keypoints..." << std::endl;
    this->matcher->matchKeypoints( frame1, frame2 );


    auto match_end_time = high_resolution_clock::now();             // Timer


    // ==================================== 
    //      Relative pose calculation
    // ==================================== 
    if (this->isShowLog()) std::cout << "Computing relative pose..." << std::endl;
    cv::Mat img_copy = img.clone();
    shared_ptr<Pose> rel_pose = this->pose_calculator->calculate( frame1, frame2, img_copy );

    if (rel_pose == nullptr)
    {
        return 0;
    }
    
    rel_pose->updateParametrization(this->pose_param);
    this->updateGlobalPose(rel_pose->getTMatrix(), frame1);

    auto rel_pose_calc_end_time = high_resolution_clock::now();     // Timer


    // ==================================== 
    //              Map update
    // ==================================== 
    if (this->isShowLog()) std::cout << "Updating map..." << std::endl;
    if (rel_pose != nullptr)
    {
        this->map_point_reg->registerMP( frame1, frame2, this->getMap3D() );
    }


    auto map_update_end_time = high_resolution_clock::now();        // Timer


    if (rel_pose != nullptr)
    {
        this->appendTrackingFrame(frame1);
    }

    // ====================================
    //              Cleanup
    // ====================================
    if (this->isShowLog()) std::cout << "Cleaning up..." << std::endl;
    this->frameListPruning();

    auto cleanup_end_time = high_resolution_clock::now();           // Timer


    if (this->isShowLog())
    {
        if (rel_pose != nullptr)
        {
            std::cout << "Parametrization: \n" << *rel_pose->getParametrization(this->pose_param) << std::endl;
        }
        std::cout << "Global Pose: \n" << this->frame_list[this->getFrameListLength()-1]->getGlobalPose() << std::endl;
    }

    if (this->show_timings)
    {
        auto ms0 = duration_cast<milliseconds>(kpts_start_time-preprocess_start_time);
        auto ms1 = duration_cast<milliseconds>(kpts_end_time-kpts_start_time);
        auto ms2 = duration_cast<milliseconds>(match_end_time-kpts_end_time);
        auto ms3 = duration_cast<milliseconds>(rel_pose_calc_end_time-match_end_time);
        auto ms4 = duration_cast<milliseconds>(map_update_end_time-rel_pose_calc_end_time);
        auto ms5 = duration_cast<milliseconds>(cleanup_end_time-map_update_end_time);
        std::cout << "Frame preprocessing time: " << ms0.count() << "ms" << std::endl;
        std::cout << "Keypoint calculation time: " << ms1.count() << "ms" << std::endl;
        std::cout << "Matching time: " << ms2.count() << "ms" << std::endl;
        std::cout << "Relalive pose calculation time: " << ms3.count() << "ms" << std::endl;
        std::cout << "Map update time: " << ms4.count() << "ms" << std::endl;
        std::cout << "Cleanup time: " << ms5.count() << "ms" << std::endl;
    }
    return 1;
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

void FTracker::drawKeypointTrails(cv::Mat &img, int frame_nr, int trail_thickness)
{
    /* Prints a line from the matched keypoints from frame nr <frame_nr> 
       to <frame_nr - max(trail_length, tracking_window_length)>
       Assumption: given more match possibilites optimal forward match 
       is equal to optimal backwards match, and that matches of subsequent
       frames are always available */

    int trail_length = this->kpt_trail_length;

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

    E_matrix = curr_frame->getRelPose( curr_frame->getFrameNr()-1 )->getEMatrix(); //TODO: Should not be 1, depends on <comparion frame spacing variable>
    F_matrix = fundamentalFromEssential( E_matrix, curr_frame->getKMatrix() );
    epipole = calculateEpipole( F_matrix );
    if (isnan(epipole.at<double>(2,0)))
    {
        epipole = (cv::Mat_<double>(3,1) << curr_frame->getImg().cols/2, 
                                            curr_frame->getImg().rows/2, 
                                            1);
    }
    drawCircle( img_disp, epipole );
}

void FTracker::drawEpipolarLinesWithPrev(cv::Mat &img_disp, int frame_nr)
{
    //TODO: Change variable <frame_nr> to <frame_idx> as frame_nr has secondary meaning
    vector<cv::Point2f> pts1, pts2;
    cv::Mat E_matrix, F_matrix, epipole;
    std::shared_ptr<FrameData> curr_frame, prev_frame;

    if (frame_nr == -1)
    {
        frame_nr = this->getFrameListLength()-1;
    }
    curr_frame = this->getFrame( frame_nr );
    prev_frame = this->getFrame( frame_nr - 1 );    //TODO: Should not be 1, depends on <comparion frame spacing variable>
    pts1 = curr_frame->compileCV2DPoints();
    pts2 = prev_frame->compileCV2DPoints();
    E_matrix = curr_frame->getRelPose( prev_frame->getFrameNr() )->getEMatrix(); 
    F_matrix = fundamentalFromEssential( E_matrix, curr_frame->getKMatrix() );
    drawEpipolarLines( F_matrix, img_disp, pts2, pts1 );
}

void FTracker::analysis(cv::Mat& img_disp)
{
    if (this->is_pose_analysis == true)
    {
        this->pose_calculator->analysis(img_disp, this->getFrame(-1), this->getFrame(-2)); //Change this to be able to handle other than the last two frames
    }
}

void FTracker::kptMatchAnalysisWithPrev( cv::Mat &img_disp, int frame_idx )
{
    int random_idx, canvas_h, canvas_w;
    double hamming_dist;
    cv::Mat img1, img2, F_matrix;
    cv::Mat canvas(800, 1400, img_disp.type(), cv::Scalar::all(0));
    shared_ptr<FrameData> frame1, frame2;
    shared_ptr<KeyPoint2> kpt1, kpt2;
    vector<shared_ptr<KeyPoint2>> matched_kpts1, matched_kpts2;

    if (frame_idx == -1)
    {
        frame_idx = this->getFrameListLength() - 1;  //TODO: Should not be 1, depends on <comparion frame spacing variable>
    }

    copyMakeBorder(img_disp, canvas, 0, canvas.rows-img_disp.rows, 0, canvas.cols-img_disp.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0) );

    frame1 = this->getFrame( frame_idx );
    frame2 = this->getFrame( frame_idx -1 );

    img1 = frame1->getImg();
    img2 = frame2->getImg();
    cvtColor(img1, img1, cv::COLOR_GRAY2BGR );
    cvtColor(img2, img2, cv::COLOR_GRAY2BGR );
    
    matched_kpts1 = frame1->getMatchedKeypoints( frame2->getFrameNr() );
    matched_kpts2 = frame2->getMatchedKeypoints( frame1->getFrameNr() );
    F_matrix = fundamentalFromEssential(frame1->getRelPose( frame2 )->getEMatrix(), frame1->getKMatrix());

    int border = 30;
    cv::Size size(101,101);

    //srand (time(NULL));
    for ( int i = 0; i < 10; ++i )
    {
        random_idx = rand() % matched_kpts1.size();
        kpt1 = matched_kpts1[random_idx];
        kpt2 = matched_kpts2[random_idx];
        KeyPoint2::drawEnchancedKeyPoint( canvas, img2, kpt2, cv::Point((border + size.width)*i, 400), size, cv::Mat());
        //KeyPoint2::drawEnchancedKeyPoint( canvas, img1, kpt1, cv::Point((border + size.width)*i, 400), cv::Size(31,31), F_matrix, kpt2);
        KeyPoint2::drawEnchancedKeyPoint( canvas, img1, kpt1, cv::Point((border + size.width)*i, 510), size, F_matrix, kpt2);
        hamming_dist = cv::norm(kpt1->getDescriptor("orb"), kpt2->getDescriptor("orb"), cv::NORM_HAMMING);
        drawIndicator(canvas, 100*(255 - 2*hamming_dist) / 255, cv::Point((border + size.width)*i, 620));
    }

    cv::imshow("KeyPoint Analysis", canvas);
    cv::waitKey(0);
}


void FTracker::incremental3DMapTrackingLog(shared_ptr<FrameData> frame, string ILog)
{
    /*
    Arguments:
        frame:  Latest tracking frame
    Overview:
        If frame is keyframe save all relevant Frame data to json file
    Note:
        TODO: Check if frame is keyframe
    */

    json data;

    //Extract data from frame
    Mat T_wc = frame->getGlobalPose();
    data["frame_nr"] = frame->getFrameNr();
    data["img_id"] = frame->getImgId();
    data["T_wc"] = {{T_wc.at<double>(0,0), T_wc.at<double>(0,1), T_wc.at<double>(0,2), T_wc.at<double>(0,3)},
                    {T_wc.at<double>(1,0), T_wc.at<double>(1,1), T_wc.at<double>(1,2), T_wc.at<double>(1,3)},
                    {T_wc.at<double>(2,0), T_wc.at<double>(2,1), T_wc.at<double>(2,2), T_wc.at<double>(2,3)}};
    vector<shared_ptr<KeyPoint2>> kpt_list = frame->getKeypoints();

    shared_ptr<MapPoint> map_point;
    vector<vector<double>> kpt_loc_list;
    vector<int> map_point_id_list;
    vector<vector<double>> map_point_loc_list;
    vector<vector<double>> map_point_unc_list;
    for ( shared_ptr<KeyPoint2> kpt : kpt_list )
    {
        map_point = kpt->getMapPoint();
        
        if (map_point != nullptr)
        {
            // Extract data from keypoint
            kpt_loc_list.push_back(vector{kpt->getCoordX(), kpt->getCoordY()});

            // Extract data from map point
            map_point_id_list.push_back(map_point->getId());
            map_point_loc_list.push_back(vector{{map_point->getCoordX(), map_point->getCoordY(), map_point->getCoordZ()}});
            map_point_unc_list.push_back({map_point->getSTDX(), map_point->getSTDY(), map_point->getSTDZ()});
        }
    }

    data["KP_loc"] = kpt_loc_list;
    data["MP_id"] = map_point_id_list;
    data["MP_loc"] = map_point_loc_list;
    data["MP_unc"] = map_point_unc_list;


    string file_path = ILog + std::to_string(frame->getFrameNr()) + ".json";
    std::ofstream file(file_path);
    file << data.dump(4);
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