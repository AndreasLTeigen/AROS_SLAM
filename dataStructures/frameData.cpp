#include <map>
#include <omp.h>
#include <shared_mutex>
#include <opencv2/opencv.hpp>

#include "frameData.hpp"

#define THREAD_NUM 4

using std::map;
using std::vector;
using std::shared_ptr;
using cv::Mat;
using cv::DMatch;

/*FrameData::FrameData(int frame_nr, int img_id, Mat K_matrix)
{
    //this->frame_nr=frame_nr;
    //this->img_id = img_id;
    this->setKMatrix( K_matrix );
}*/

FrameData::~FrameData()
{
    std::unique_lock lock2(this->mutex_kpts);
    //std::cout << "FRAME DESTROYED: " << this->frame_nr << std::endl;
    this->kpts.clear();
}

// ----------- Write functions -------------

void FrameData::setImg( Mat &img )
{
    std::unique_lock lock(this->mutex_img);
    this->img = img;
}

void FrameData::setKMatrix( Mat K_matrix )
{
    std::unique_lock lock(this->mutex_K_matrix);
    this->K_matrix = K_matrix;
}

void FrameData::setGlobalPose( cv::Mat global_pose)
{
    std::unique_lock lock(this->mutex_global_pose);
    this->global_pose = global_pose;
}

void FrameData::setAllKeypoints( vector<std::shared_ptr<KeyPoint2>> kpts )
{
    std::unique_lock lock(this->mutex_kpts);
    this->kpts = kpts;
}

void FrameData::promoteToKeyframe()
{
    std::unique_lock lock(this->mutex_is_keyframe);
    this->is_keyframe = true;
}

void FrameData::demoteFromKeyframe()
{
    std::unique_lock lock(this->mutex_is_keyframe);
    this->is_keyframe = false;
}

void FrameData::addKeypoint(shared_ptr<KeyPoint2> kpt)
{
    std::unique_lock lock(this->mutex_kpts);
    this->kpts.push_back(kpt);
}

void FrameData::registerKeypoints(vector<shared_ptr<KeyPoint2>> kpts)
{
    /* Registers vector<shared_ptr<KeyPoint2>> and saves it in the frameData class. 
    For more details, see design document*/

    #pragma omp parallel for
    for( shared_ptr<KeyPoint2> kpt : kpts )
    {
        this->addKeypoint(kpt);
    }
}

void FrameData::registerKeypoints(vector<cv::KeyPoint>& kpts, Mat& descrs)
{
    /* Converts and registers vector<cv::KeyPoint> into the AVG Keypoint class
       and saves it in the frameData class. For more details, see design document*/

    #pragma omp parallel for
    for( int i = 0; i < kpts.size(); i++ )
    {
        shared_ptr<KeyPoint2> keypoint = std::make_shared<KeyPoint2>(i, kpts[i], this->getFrameNr(), descrs.row(i));
        this->addKeypoint(keypoint);
        //std::cout << "NUm threads: " << omp_get_num_threads() << std::endl;
        //std::cout << "This is thread num: " << omp_get_thread_num() << std::endl;
    }
}

void FrameData::removeMatchedKeypointsByIdx(int matched_frame_nr, vector<int> kpt_idx_list)
{
    /*
    Arguments:
        matched_frame_nr:                       Frame nr of frame connecting matches of interest to <this> 
                                                frame.
        kpt_idx_list:                           Index of keypoints to be removed from <this> frame's 
                                                <matched_kpts> list.
    Effect:
        this->matched_kpts[matched_frame_nr]:   Removes keypoints from list corresponding to the indexes 
                                                in <kpt_idx_list>
    */
    // Removes the entries from the matched keypoint list with frame <mathed_frame_nr> given by index entry in <kpt_idx_list>
    std::unique_lock lock(this->mutex_matched_kpts);

    for (int i = kpt_idx_list.size()-1; i >= 0; i--)
    {
        this->matched_kpts[matched_frame_nr].erase(this->matched_kpts[matched_frame_nr].begin()+kpt_idx_list[i]);
    }
}

vector<int> FrameData::removeOutlierMatches(cv::Mat inliers, shared_ptr<FrameData> connecting_frame)
{
    /*
    Arguments:
        inliers:                            Mask of inliers/outliers between the ordered lists of 
                                            <matched_kpts[frameX]> of <this> frame and <connecting_frame>.
        connecting_frame:                   Frame object connecting matches of interest to <this> frame.
    Effect:
        frameX->kpts[Z]->matches[frameY]:   Deletes match object between keypoints in <frame1> and <frame2> 
                                            corresponding to <inliers> mask.
    */

    shared_ptr<KeyPoint2> kpt1;
    vector<int> removed_matched_keypoint_idx;

    std::shared_lock lock(this->mutex_matched_kpts);

    vector<shared_ptr<KeyPoint2>> matched_kpt_list = this->matched_kpts[connecting_frame->getFrameNr()];

    #pragma omp parallel for
    for (int i = 0; i < matched_kpt_list.size(); i++)
    {
        if(!inliers.at<uchar>(i))
        {
            kpt1 = matched_kpt_list[i];
            kpt1->removeAllMatches(connecting_frame->getFrameNr());


            removed_matched_keypoint_idx.push_back(i);
        }
    }

    return removed_matched_keypoint_idx;
}

std::vector<int> FrameData::removeMatchesWithLowConfidence(double threshold, std::shared_ptr<FrameData> connecting_frame)
{
    /*
    Arguments:
        threshold:                          Match confidence threshold.
        connecting_frame:                   Frame object connecting matches of interest to <this> frame.
    Effect:
        frameX->kpts[Z]->matches[frameY]:   Deletes match object between keypoints in <frame1> and <frame2> 
                                            whose confidence is lower than <threshold>.
    */

    std::shared_lock lock(this->mutex_matched_kpts);

    shared_ptr<KeyPoint2> kpt1;
    shared_ptr<Match> match1;
    vector<int> removed_matched_keypoint_idx;

    vector<shared_ptr<KeyPoint2>> matched_kpt_list = this->matched_kpts[connecting_frame->getFrameNr()];

    #pragma omp parallel for
    for (int i = 0; i < matched_kpt_list.size(); i++)
    {
        kpt1 = matched_kpt_list[i];
        match1 = kpt1->getHighestConfidenceMatch(connecting_frame->getFrameNr());
        if(match1->getConfidence() < threshold)
        {
            kpt1->removeAllMatches(connecting_frame->getFrameNr());
            removed_matched_keypoint_idx.push_back(i);
        }
    }
    return removed_matched_keypoint_idx;
}

void FrameData::addKptToMatchList(shared_ptr<KeyPoint2> kpt, shared_ptr<FrameData> connecting_frame)
{
    /*
    Argument:
        kpt:                                    Keypoint of interest from <this> frame.
        connecting_frame:                       Frame object connecting match of interest to <this> frame.
    Effect:
        this->matched_kpts[connecting_frame]:   Adds <kpt> to list of matched keypoints with <connecting_frame>.
    */
    std::unique_lock lock(this->mutex_matched_kpts);
    this->matched_kpts[connecting_frame->getFrameNr()].push_back(kpt);
}

void FrameData::addRelPose(shared_ptr<Pose> rel_pose, shared_ptr<FrameData> connecting_frame)
{
    /*
    Arguments:
        rel_pose:                           Relative <Pose> object between <this> frame and <connecting_frame>
        connecting_frame:                   Frame object connecting <this> frame to the relative pose of interest.
    Effect:
        this->rel_poses[connecting_frame]:  Registers the pose in <this> frame
    */
    std::unique_lock lock(this->mutex_rel_poses);
    this->rel_poses[connecting_frame->getFrameNr()] = rel_pose;
}

// ----------- Static write functions -------------

void FrameData::registerMatches(shared_ptr<FrameData> frame1, shared_ptr<FrameData> frame2, vector<vector<DMatch>>& matches)
{
    /* 
    Arguments:
        frameX:                             Frames between whose matches are of interest.
        matches:                            List of matches given in the OpenCV style of <vector<vector<DMatch>>>.
    Effect:
        frameX->kpts[Z]->matches[frameY]:   Register new <Match> objects in list based on information in <matches>.
        frameX->matched_kpts[frameZ]:       Matched keypoints are registered in list for efficiency sake.
    Note:                                   <frame1> and <frame2> needs to be in the same order as they were
                                            given in the matching process.
    TODO:                                   Fix requirement for note
    */

    vector<cv::DMatch> kpt_match;
    shared_ptr<KeyPoint2> kpt1, kpt2;
    
    for(int i = 0; i < matches.size(); i++)
    {
        kpt_match = matches[i];
        for (int j = 0; j < matches[i].size(); j++)
        {
            
            kpt1 = frame1->kpts[kpt_match[j].queryIdx];
            kpt2 = frame2->kpts[kpt_match[j].trainIdx];

            shared_ptr<Match> match = shared_ptr<Match>(new Match(kpt1, kpt2, kpt_match[j].distance, i));
            kpt1->addMatch(match, frame2->getFrameNr());
            kpt2->addMatch(match, frame1->getFrameNr());

            frame1->addKptToMatchList(kpt1, frame2);
            frame2->addKptToMatchList(kpt2, frame1);

        }
    }
}

shared_ptr<Pose> FrameData::registerRelPose(Mat E_matrix, shared_ptr<FrameData> frame1, shared_ptr<FrameData> frame2)
{
    /*
    Arguments:
        E_matrix:                   Essential matrix of the transformation between <frame1> and <frame2>
        frameX:                     Frame objects connecting the relative pose of interest.
    Returns:
        rel_pose:                   Newly created relative pose object between <frame1> and <frame2>.
    Effect:
        frameX->rel_poses[frameY]:  Registers relative poses between <frame1> and <frame2> based on <E_matrix>.
    */
    shared_ptr<Pose> rel_pose = shared_ptr<Pose>(new Pose(E_matrix, frame1, frame2));
    frame1->addRelPose(rel_pose, frame2);
    frame2->addRelPose(rel_pose, frame1);
    return rel_pose;
}

shared_ptr<Pose> FrameData::registerGTRelPose(Mat T_matrix, shared_ptr<FrameData> frame1, shared_ptr<FrameData> frame2)
{
    /*
    Arguments:
        T_matrix:                   Relative transformation matrix between <frame1> and <frame2>
        frameX:                     Frame objects connecting the relative pose of interest.
    Returns:
        rel_pose:                   Newly created relative pose object between <frame1> and <frame2>.
    Effect:
        frameX->rel_poses[frameY]:  Registers relative poses between <frame1> and <frame2> based on <E_matrix>.
    */
    shared_ptr<Pose> rel_pose = shared_ptr<Pose>(new Pose(frame1, frame2));
    rel_pose->updatePoseVariables(T_matrix);
    frame1->addRelPose(rel_pose, frame2);
    frame2->addRelPose(rel_pose, frame1);
    return rel_pose;
}

void FrameData::removeOutlierMatches(cv::Mat inliers, shared_ptr<FrameData> frame1, shared_ptr<FrameData> frame2)
{
    /*
    Arguments:
        inliers:                            Mask of inliers/outliers between the ordered lists of 
                                            <matched_kpts[frameX]> of both frames.
        frameX:                             Frames between whose matches are of interest.
    Effect:
        frameX->kpts[Z]->matches[frameY]:   Deletes match object between keypoints in <frame1> and <frame2> 
                                            corresponding to <inliers> mask.
        frameX->matched_kpts[frameY]:       Removes keypoints in <matched_kpts[frameY]> lists of both frames 
                                            corresponding to the matches removed.
    TODO:                                   Make this into a friend function such that both <mutex_matched_kpts>
                                            can be locked before starting to remove anything
    */
    
    // Remove actuall match between keypoints
    vector<int> removed_matched_keypoint_idx = frame1->removeOutlierMatches(inliers, frame2);

    // Remove keypoints from <matched_kpts> lists
    frame1->removeMatchedKeypointsByIdx(frame2->getFrameNr(), removed_matched_keypoint_idx);
    frame2->removeMatchedKeypointsByIdx(frame1->getFrameNr(), removed_matched_keypoint_idx);
}

void FrameData::removeMatchesWithLowConfidence( double threshold, shared_ptr<FrameData> frame1, shared_ptr<FrameData> frame2 )
{
    /*
    Arguments:
        threshold:      Match confidence threshold.
        frameX:         Frames between whose matches are of interest.
    Effect:
        frameX->kpts[Z]->matches[frameY]:   Deletes match object between keypoints in <frame1> and <frame2>
                                            whose confidence is lower than <threshold>.
        frameX->matched_kpts[frameY]:       Removes keypoints in <matched_kpts[frameY] lists of both frames
                                            corresponding to the matches removed.
    */
    
    // Remove actuall match between keypoints
    vector<int> removed_matched_keypoint_idx = frame1->removeMatchesWithLowConfidence(threshold, frame2);

    // Remove keypoints from <matched_kpts> lists
    frame1->removeMatchedKeypointsByIdx(frame2->getFrameNr(), removed_matched_keypoint_idx);
    frame2->removeMatchedKeypointsByIdx(frame1->getFrameNr(), removed_matched_keypoint_idx);
}


// ----------- Read functions -------------
bool FrameData::isKeyframe()
{
    /*
    Returns:
        this->is_keyframe:  <True> if <this> frame is keyframe.
    */
    std::shared_lock lock(this->mutex_is_keyframe);
    return this->is_keyframe;
}

int FrameData::getFrameNr()
{
    /*
    Returns:
        this->frame_nr:     Frame number of current frame, unique per frame.
    */
    return this->frame_nr;
}

int FrameData::getImgId()
{
    /*
    Returns:
        this->img_id:       Image id of current frame, unique per image.
    */
    return this->img_id;
}

int FrameData::getNumKeypoints()
{
    std::shared_lock lock(this->mutex_kpts);
    return this->kpts.size();
}

Mat FrameData::getImg()
{
    /*
    Returns:
        this->img:     image of <this> frame.
    */
    std::shared_lock lock(this->mutex_img);
    return this->img.clone();
}

Mat FrameData::getKMatrix()
{
    /*
    Returns:
        this->K_matrix:     Kamera calibration matrix of <this> frame.
    */
    std::shared_lock lock(this->mutex_K_matrix);
    return this->K_matrix.clone();
}

Mat FrameData::getGlobalPose()
{
    /*
    Returns:
        this->global_pose:  Pose of <this> frame in reference to global map.
    */
    std::shared_lock lock(this->mutex_global_pose);
    return this->global_pose.clone();
}

vector<shared_ptr<KeyPoint2>> FrameData::getKeypoints()
{
    /*
    Returns:
        this->kpts:         List of keypoints observed in <this> frame.
    */
    std::shared_lock lock(this->mutex_kpts);
    return this->kpts;
}

vector<shared_ptr<KeyPoint2>> FrameData::getMatchedKeypoints(int matched_frame_nr)
{
    /* 
    Arguments:
        rel_frame_nr:       Frame nr of connecting frame.
    Returns:
        matched_keypoints:  List of keypoints from <this> frame containing all keypoints that are 
                            mached with frame nr <rel_frame_nr>.
    */
    std::shared_lock lock(this->mutex_matched_kpts);
    return this->matched_kpts[matched_frame_nr];
}

shared_ptr<Pose> FrameData::getRelPose(int rel_frame_nr)
{
    /* 
    Arguments:
        rel_frame_nr:   Frame nr of connecting frame with relative pose of interest.
    Returns:
        rel_pose:       Pose object of relative pose between <this> frame and frame nr <rel_frame_nr>.
    */
    
    std::shared_lock lock(this->mutex_rel_poses);
    if (this->rel_poses.find(rel_frame_nr) == this->rel_poses.end())
    {
        std::cout << "Relative pose not registered" << std::endl;
    }
    return rel_poses[rel_frame_nr];
}

shared_ptr<Pose> FrameData::getRelPose( shared_ptr<FrameData> rel_frame )
{
    /* 
    Arguments:
        rel_frame:      Frame object of connecting frame with relative pose of interest.
    Returns:
        rel_pose:       Pose object of relative pose between <this> frame and <rel_frame>.
    */
    return this->getRelPose( rel_frame->getFrameNr() );
}

vector<cv::KeyPoint> FrameData::compileCVKeypoints()
{
    /* 
    Returns:
        kpts_cv:    All keypoints of <this> frame in the OpenCV <KeyPoint> class format.

    TODO: Can possibly be made more efficient by allocating the full vector first since number of keypoints are known. 
    */

    std::shared_lock lock(this->mutex_kpts);

    vector<shared_ptr<KeyPoint2>> kpts = this->getKeypoints();

    return this->compileCVKeypoints( kpts );
}

Mat FrameData::compileCVDescriptors(std::string descr_type)
{
    /* 
    Returns: 
        descrs_cv:  Descriptors of all keypoints in frame in the OpenCV format.

    TODO: If there are no keypoints registered in the frame, function
       will crash.
    */

    std::shared_lock lock(this->mutex_kpts);

    std::shared_ptr<KeyPoint2> temp_kpt;
    int N = this->getKeypoints().size();
    temp_kpt = this->getKeypoints()[0];
    int len_descr = temp_kpt->getDescriptor(descr_type).cols;
    int descr_value_type = temp_kpt->getDescriptor(descr_type).type();

    Mat descrs_cv = Mat::zeros(N, len_descr, descr_value_type);
    
    #pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
        temp_kpt = this->getKeypoints()[i];
        temp_kpt->getDescriptor(descr_type).copyTo(descrs_cv.row(i));
    }

    return descrs_cv;
}

vector<cv::Point2f> FrameData::compileCV2DPoints()
{
    /*
    Returns: 
        List of points in the <cv::Point> format corresponding to the keypoints in <this> frame.
    */
    
    std::shared_lock lock(this->mutex_kpts);
    return FrameData::compileCV2DPointsN(this->getKeypoints());
}


cv::Mat FrameData::compileMatchedCVPointCoords( int matched_frame_nr)
{
    /*
    Arguments:  
        matched_frame_nr:   Frame nr of connecting frame with matches of interest.
    Returns:    
        uv:                 Matrix containing the matched keypoints in homogeneous pixel corrdinates 
                            of "this" frame [shape 3 x n].
    */

    std::shared_lock lock(this->mutex_matched_kpts);

    vector<shared_ptr<KeyPoint2>> matched_kpts = this->matched_kpts[matched_frame_nr];
    /*
    Mat uv(3,matched_kpts.size(), CV_64F);
    shared_ptr<KeyPoint2> kpt;

    #pragma omp parallel for
    for (int i = 0; i < matched_kpts.size(); i++)
    {
        kpt = matched_kpts[i];
        uv.at<double>(0,i) = kpt->getCoordX();
        uv.at<double>(1,i) = kpt->getCoordY();
        uv.at<double>(2,i) = 1.0;
    }
    return uv;
    */
    return compileCVPointCoords( matched_kpts );
}

std::vector<cv::Point> FrameData::compileMatchedCVPoints( int matched_frame_nr )
{
    /*
    Arguments:
        matched_frame_nr:   Frame nr of connecting frame with matches of interest.
    Returns:
        matched_points:     Vector containing the matched points of "this" frame in a
                            cv::Points format.
    */
    std::shared_lock lock(this->mutex_matched_kpts);

    shared_ptr<KeyPoint2> kpt;
    vector<shared_ptr<KeyPoint2>> matched_kpts = this->matched_kpts[matched_frame_nr];
    vector<cv::Point> matched_points(matched_kpts.size());

    #pragma omp parallel for
    for (int i = 0; i < matched_kpts.size(); i++)
    {
        kpt = matched_kpts[i];
        matched_points[ i ] = kpt->compileCV2DPoint();
    }
    return matched_points;
}


// ----------- Static read functions -------------
cv::Mat FrameData::compileCVPointCoords( std::vector<std::shared_ptr<KeyPoint2>> kpts )
{
    //TODO: This function does not return pure integer values but rather changes them into inprecice double values, fix this
    Mat uv(3,kpts.size(), CV_64F);
    shared_ptr<KeyPoint2> kpt;

    #pragma omp parallel for
    for (int i = 0; i < kpts.size(); i++)
    {
        kpt = kpts[i];
        uv.at<double>(0,i) = kpt->getCoordX();
        uv.at<double>(1,i) = kpt->getCoordY();
        uv.at<double>(2,i) = 1.0;
    }
    return uv;
}

vector<cv::Point2f> FrameData::compileCV2DPointsN(vector<shared_ptr<KeyPoint2>> kpts)
{
    /*
    Arguments:
        kpts:           List of keypoints.
    Returns:
        points2D_cv:    List of points in the <cv::Point2f> format corresponding to the kpts list.
    */
    vector<cv::Point2f> points2D_cv(kpts.size());
    for (int i = 0; i < kpts.size(); i++)
    {
        points2D_cv[i] = cv::Point2f(kpts[i]->compileCV2DPoint());
    }
    return points2D_cv;
}

std::vector<cv::KeyPoint> FrameData::compileCVKeypoints( std::vector<std::shared_ptr<KeyPoint2>> kpts )
{
    // TODO: Move the inner part of this for loop to the KeyPoint2 function
    int N = kpts.size();
    vector<cv::KeyPoint> kpts_cv( N );

    #pragma omp parallel for
    for ( int i = 0; i < N; i++ )
    {
        shared_ptr<KeyPoint2> kpt = kpts[i];
        cv::KeyPoint* new_kpt = new cv::KeyPoint(kpt->getCoordX(), kpt->getCoordY(),
                                            kpt->getSize(), kpt->getAngle(),
                                            kpt->getResponse(), kpt->getOctave());
        kpts_cv[i] = *new_kpt;
    }
    return kpts_cv;
}


// ----------- Friend functions -------------
void compileMatchedCVPointCoords(std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2, cv::Mat& frame1_points, cv::Mat& frame2_points )
{
    /*
    Arguments:
        frameX:         Frame 1 and 2 that connects the matches of interest.
    Returns:
        frameX_points:  Matrix containing the matched coordinates in homogeneous pixel coordinates 
                        for the respective frame [shape 3 x n].
    */

    std::shared_ptr<FrameData> newest_frame, oldest_frame;
    cv::Mat newest_frame_points, oldest_frame_points;

    // Sorting the frames to avoid mutex softlock.
    if ( frame1->getFrameNr() > frame2->getFrameNr() )
    {
        newest_frame = frame1;
        oldest_frame = frame2;
    }
    else
    {
        newest_frame = frame2;
        oldest_frame = frame1;
    }

    std::shared_lock lock1(newest_frame->mutex_matched_kpts);
    std::shared_lock lock2(oldest_frame->mutex_matched_kpts);

    newest_frame_points = newest_frame->compileMatchedCVPointCoords( oldest_frame->getFrameNr() );
    oldest_frame_points = oldest_frame->compileMatchedCVPointCoords( newest_frame->getFrameNr() );

    // Assigns the right point groups to the corresponding original frame.
    if ( frame1->getFrameNr() > frame2->getFrameNr() )
    {
        frame1_points = newest_frame_points;
        frame2_points = oldest_frame_points;
    }
    else
    {
        frame2_points = newest_frame_points;
        frame1_points = oldest_frame_points;
    }
}

void compileMatchedCVPoints(std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2, std::vector<cv::Point>& frame1_points, std::vector<cv::Point>& frame2_points)
{
    /*
    Arguments:
        frameX:         Frame 1 and 2 that connects the matches of interest.
    Returns:
        framex_points:  Vector containing the matched points of the respecitve frames
                        in cv::Points format.
    */
    
    shared_ptr<FrameData> newest_frame, oldest_frame;
    vector<cv::Point> newest_frame_points, oldest_frame_points;

    // Sorting the frames to avoid mutex softlock.
    if ( frame1->getFrameNr() > frame2->getFrameNr() )
    {
        newest_frame = frame1;
        oldest_frame = frame2;
    }
    else
    {
        newest_frame = frame2;
        oldest_frame = frame1;
    }

    std::shared_lock lock1(newest_frame->mutex_matched_kpts);
    std::shared_lock lock2(oldest_frame->mutex_matched_kpts);

    newest_frame_points = newest_frame->compileMatchedCVPoints( oldest_frame->getFrameNr() );
    oldest_frame_points = oldest_frame->compileMatchedCVPoints( newest_frame->getFrameNr() );

    // Assigns the right point groups to the corresponding original frame.
    if ( frame1->getFrameNr() > frame2->getFrameNr() )
    {
        frame1_points = newest_frame_points;
        frame2_points = oldest_frame_points;
    }
    else
    {
        frame2_points = newest_frame_points;
        frame1_points = oldest_frame_points;
    }
}

void copyMatchedKptsLists(shared_ptr<FrameData> frame1,shared_ptr<FrameData> frame2, vector<shared_ptr<KeyPoint2>>& frame1_matched_kpts, vector<shared_ptr<KeyPoint2>>& frame2_matched_kpts )
{
    /*
    Arguments:
        frameX:                 Frames of which <matched_kpts[frameY]> should be copied.
    Returns:
        frameX_matched_kpts:    Copied matched keypoint lists <matched_kpts[frameY]> that is guaranteed to be
                                thread safe and synchronized.
    */

    shared_ptr<FrameData> newest_frame, oldest_frame;
    vector<shared_ptr<KeyPoint2>> newest_frame_matched_points, oldest_frame_matched_points;
    
    // Sorting the frames to avoid mutex softlock.
    if ( frame1->getFrameNr() > frame2->getFrameNr() )
    {
        newest_frame = frame1;
        oldest_frame = frame2;
    }
    else
    {
        newest_frame = frame2;
        oldest_frame = frame1;
    }

    std::shared_lock lock1(newest_frame->mutex_matched_kpts);
    std::shared_lock lock2(oldest_frame->mutex_matched_kpts);

    newest_frame_matched_points = newest_frame->getMatchedKeypoints( oldest_frame->getFrameNr() );
    oldest_frame_matched_points = oldest_frame->getMatchedKeypoints( newest_frame->getFrameNr() );

    // Assigns the right point groups to the corresponding original frame.
    if ( frame1->getFrameNr() > frame2->getFrameNr() )
    {
        frame1_matched_kpts = newest_frame_matched_points;
        frame2_matched_kpts = oldest_frame_matched_points;
    }
    else
    {
        frame2_matched_kpts = newest_frame_matched_points;
        frame1_matched_kpts = oldest_frame_matched_points;
    }
}