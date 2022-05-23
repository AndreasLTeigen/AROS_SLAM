#ifndef frameData_h
#define frameData_h

#include "pose.hpp"
#include "keypoint.hpp"

#include <shared_mutex>
#include <opencv2/opencv.hpp>

// Forward declaration for circular dependence
class Pose;

class FrameData
{
    private:
        // Variables
        const int frame_nr, img_id;
        int n_keypoints;
        bool is_keyframe = false;
        cv::Mat img;
        cv::Mat K_matrix;
        cv::Mat global_pose = cv::Mat::eye(4,4,CV_64F);
        std::vector<std::shared_ptr<KeyPoint2>> kpts;
        std::map<int, std::shared_ptr<Pose>> rel_poses;
        std::map<int, std::vector<std::shared_ptr<KeyPoint2>>> matched_kpts;

        // Mutexes
        mutable std::shared_mutex mutex_img;
        mutable std::shared_mutex mutex_K_matrix;
        mutable std::shared_mutex mutex_global_pose;
        mutable std::shared_mutex mutex_is_keyframe;
        mutable std::shared_mutex mutex_kpts;
        mutable std::shared_mutex mutex_rel_poses;
        mutable std::shared_mutex mutex_matched_kpts;

    public:
        FrameData( int frame_nr, int img_id, cv::Mat K_matrix ) : frame_nr(frame_nr), img_id(img_id)
                                                                                {this->setKMatrix( K_matrix );} ;
        ~FrameData();

        // Write functions
        void setImg( cv::Mat& img );
        void setKMatrix( cv::Mat K_matrix );
        void setGlobalPose( cv::Mat global_pose);
        void setAllKeypoints( std::vector<std::shared_ptr<KeyPoint2>> kpts );
        void promoteToKeyframe();
        void demoteFromKeyframe();
        void addKeypoint( std::shared_ptr<KeyPoint2> kpt );
        void registerKeypoints(std::vector<std::shared_ptr<KeyPoint2>> kpts);
        void registerKeypoints( std::vector<cv::KeyPoint>& kpts, cv::Mat& descrs );
        void removeMatchedKeypointsByIdx( int matched_frame_nr, std::vector<int> kpt_idx_list );
        std::vector<int> removeOutlierMatches( cv::Mat inliers, std::shared_ptr<FrameData> connecting_frame );
        std::vector<int> removeMatchesWithLowConfidence(double threshold, std::shared_ptr<FrameData> connecting_frame);
        void addKptToMatchList( std::shared_ptr<KeyPoint2> kpt, std::shared_ptr<FrameData> connecting_frame );
        void addRelPose( std::shared_ptr<Pose> rel_pose, std::shared_ptr<FrameData> connecting_frame );
        
        static void registerMatches( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2, std::vector<std::vector<cv::DMatch>>& matches );
        static std::shared_ptr<Pose> registerRelPose( cv::Mat E_matrix, std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2 );
        static std::shared_ptr<Pose> registerGTRelPose(cv::Mat T_matrix, std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2);
        static void removeOutlierMatches( cv::Mat inliers, std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2 );
        static void removeMatchesWithLowConfidence( double threshold, std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2 );


        // Read functions
        bool isKeyframe();
        int getFrameNr();
        int getImgId();
        int getNumKeypoints();
        cv::Mat getImg();
        cv::Mat getKMatrix();
        cv::Mat getGlobalPose();
        std::vector<std::shared_ptr<KeyPoint2>> getKeypoints();
        std::vector<std::shared_ptr<KeyPoint2>> getMatchedKeypoints( int matched_frame_nr );
        std::shared_ptr<Pose> getRelPose( int rel_frame_nr );
        std::shared_ptr<Pose> getRelPose( std::shared_ptr<FrameData> rel_frame );
        std::vector<cv::KeyPoint> compileCVKeypoints();
        cv::Mat compileCVDescriptors(std::string descr_type="orb");
        std::vector<cv::Point2f> compileCV2DPoints();
        cv::Mat compileMatchedCVPointCoords( int matched_frame_nr);
        std::vector<cv::Point> compileMatchedCVPoints( int matched_frame_nr);
        
        static cv::Mat compileCVPointCoords( std::vector<std::shared_ptr<KeyPoint2>> kpts );
        static std::vector<cv::Point2f> compileCV2DPointsN( std::vector<std::shared_ptr<KeyPoint2>> kpts );
        static std::vector<cv::KeyPoint> compileCVKeypoints( std::vector<std::shared_ptr<KeyPoint2>> kpts );

        friend void compileMatchedCVPointCoords( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2, cv::Mat& frame1_points, cv::Mat& frame2_points );
        friend void compileMatchedCVPoints( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2, std::vector<cv::Point>& frame1_points, std::vector<cv::Point>& frame2_points );
        friend void copyMatchedKptsLists( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2, std::vector<std::shared_ptr<KeyPoint2>>& frame1_matched_kpts, std::vector<std::shared_ptr<KeyPoint2>>& frame2_matched_kpts );
};

#endif