#ifndef tracking_h
#define tracking_h

#include "../keypointExtraction/findKeypoints.hpp"
#include "../keypointMatching/matchKeypoints.hpp"
#include "../motionPrior/motionPrior.hpp"
#include "../poseCalculation/poseCalculation.hpp"
#include "../mapPointHandler/mapPointRegistration.hpp"
#include "../mapPointHandler/mapPointCulling.hpp"
#include "../dataStructures/frameData.hpp"
#include "../dataStructures/map3D.hpp"

class FTracker
{
    private:
        bool show_timings, show_tracking_log;
        cv::Mat T_global;
        Detector detec_type;
        Descriptor descr_type;
        Matcher matcher_type;
        MotionPrior motion_prior_type;
        PoseCalculator pose_calculation_type;
        PointReg3D point_reg_3D_type;
        PointCull3D point_cull_3D_type;
        int curr_frame_nr, tracking_window_length;
        std::vector<std::shared_ptr<FrameData>> frame_list;  // TODO:Change this to something like <frame_window_list>
        std::shared_ptr<Map3D> map_3d;

        // Mutexes
        mutable std::shared_mutex mutex_curr_frame_nr;
        mutable std::shared_mutex mutex_T_global;
        mutable std::shared_mutex mutex_frame_list;
        mutable std::shared_mutex mutex_map3D;

    public:
        FTracker( YAML::Node config );
        ~FTracker();

        int getCurrentFrameNr();
        int getTrackingWindowLength();
        int getFrameListLength();
        Detector getDetectorType();
        Descriptor getDescriptorType();
        MotionPrior getMotionPriorType();
        Matcher getMatcherType();
        PoseCalculator getPoseCalcuationType();
        PointReg3D getPointReg3DType();
        cv::Mat getGlobalPose();
        std::vector<std::shared_ptr<FrameData>> getTrackingFrames();
        std::shared_ptr<FrameData> getFrame(int index);
        std::shared_ptr<Map3D> getMap3D();
        void setCurrentFrameNr(int curr_frame_nr);
        void setGlobalPose(cv::Mat T_global);
        void updateGlobalPose(cv::Mat T_rel, std::shared_ptr<FrameData> current_frame);
        void initializeTracking(cv::Mat &img, int img_id, cv::Mat K_matrix, int nKeypoints=500);
        void trackFrame(cv::Mat &img, int img_id, cv::Mat K_matrix, int nKeypoints=500, int comparison_frame_spacing=1);
        void appendTrackingFrame(std::shared_ptr<FrameData> new_frame);
        void frameListPruning();
        void drawKeypoints(cv::Mat &src, cv::Mat &dst, int frame_nr=-1);
        void drawKeypointTrails(cv::Mat &img, int trail_length=49, int frame_nr=-1, int trail_thickness=2);
        void drawEpipoleWithPrev(cv::Mat &img_disp, int frame_nr1=-1);
        void drawEpipolarLinesWithPrev(cv::Mat &img_disp, int frame_nr=-1);
        void incremental3DMapTrackingLog(std::shared_ptr<FrameData> frame, std::string ILog_path);

        // Functions for error checking
        float getLongestDistanceMatch(std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2, std::shared_ptr<KeyPoint2>& kpt1, std::shared_ptr<KeyPoint2>& kpt2);
};

#endif