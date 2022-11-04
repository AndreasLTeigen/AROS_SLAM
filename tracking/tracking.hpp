#ifndef tracking_h
#define tracking_h

#include "../framePreprocessor/framePreprocessor.hpp"
#include "../keypointExtraction/keypointExtraction.hpp"
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

        std::shared_ptr<Preprocessor> frame_preprocessor;
        std::shared_ptr<MotionPrior> motion_prior;
        std::shared_ptr<Extractor> extractor;
        std::shared_ptr<Matcher> matcher;
        std::shared_ptr<PoseCalculator> pose_calculator;
        std::shared_ptr<MapPointRegistrator> map_point_reg;
        std::shared_ptr<MapPointCuller> map_point_cull;

        ParamID pose_param;

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
        cv::Mat getGlobalPose();
        //std::shared_ptr<PoseCalculator> getPoseCalculator();
        std::vector<std::shared_ptr<FrameData>> getTrackingFrames();
        std::shared_ptr<FrameData> getFrame(int index);
        std::shared_ptr<Map3D> getMap3D();
        void setCurrentFrameNr(int curr_frame_nr);
        void setGlobalPose(cv::Mat T_global);
        void updateGlobalPose(cv::Mat T_rel, std::shared_ptr<FrameData> current_frame);
        void initializeTracking(cv::Mat &img, int img_id, cv::Mat K_matrix);
        void trackFrame(cv::Mat &img, int img_id, cv::Mat K_matrix, int comparison_frame_spacing=1);
        void appendTrackingFrame(std::shared_ptr<FrameData> new_frame);
        void frameListPruning();
        void drawKeypoints(cv::Mat &src, cv::Mat &dst, int frame_nr=-1);
        void drawKeypointTrails(cv::Mat &img, int trail_length=49, int frame_nr=-1, int trail_thickness=2);
        void drawEpipoleWithPrev(cv::Mat &img_disp, int frame_nr1=-1);
        void drawEpipolarLinesWithPrev(cv::Mat &img_disp, int frame_nr=-1);
        void analysis(YAML::Node& config, cv::Mat& img_disp);
        void kptMatchAnalysisWithPrev( cv::Mat &img_disp, int frame_idx=-1 );
        void kptMatchAnalysisIterationLogWithPrev( cv::Mat &img_disp, int frame_idx=-1 );
        void incremental3DMapTrackingLog(std::shared_ptr<FrameData> frame, std::string ILog_path);

        // Functions for error checking
        float getLongestDistanceMatch(std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2, std::shared_ptr<KeyPoint2>& kpt1, std::shared_ptr<KeyPoint2>& kpt2);
};

#endif