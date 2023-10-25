#ifndef GJET_h
#define GJET_h

#include <ceres/ceres.h>

#include "../poseCalculation.hpp"
#include "../../dataStructures/frameData.hpp"
#include "../../dataStructures/keypoint.hpp"
#include "../../dataStructures/pose.hpp"
#include "../../dataStructures/parametrization.hpp"

class LossFunction
{
    protected:
        int W, H, n_reg_size;
        ParamID paramId = ParamID::LIEPARAM;

        int nfeatures = 500;
        float scaleFactor = 1.2f;
        int nlevels = 1;//8;
        int edgeThreshold = 31;//19;
        int firstLevel = 0;
        int WTA_K = 2;
        int patchSize = 31;
        int fastThreshold = 20;
        cv::Ptr<cv::ORB> orb = cv::ORB::create( nfeatures,
                                                scaleFactor,
                                                nlevels,
                                                edgeThreshold,
                                                firstLevel,
                                                WTA_K,
                                                cv::ORB::FAST_SCORE,
                                                patchSize,
                                                fastThreshold);
        // cv::Ptr<cv::ORB> orb = cv::ORB::create();
    public:
        LossFunction(cv::Mat& img);
        LossFunction(int W, int H);
        ~LossFunction(){};

        double calculated_descs = 0;

        // TODO: Move these variables to a new place
        std::string descriptor_name = "orb";
        int getImgWidth();
        int getImgHeight();
        int getPatchSize();

        virtual double calculateKptLoss(const cv::Mat& F_matrix, const cv::Mat& A_d_k, const cv::Mat& x_k, const cv::Mat& y_k, cv::Mat& v_k_opt)=0;
        virtual double calculateKptLoss(const cv::Mat& F_matrix, const std::shared_ptr<KeyPoint2> kpt1, const std::shared_ptr<KeyPoint2> kpt2, cv::Mat& v_k_opt)=0;
        virtual bool doPrecomputeDescriptors()=0;
        virtual bool validKptLoc( double x, double y, int kpt_size )=0;
        virtual bool updateKeypoint( std::shared_ptr<KeyPoint2> kpt, double x_update, double y_update )=0;
        virtual void linearizeLossFunction( cv::Mat& y_k, std::shared_ptr<KeyPoint2> kpt2, cv::Mat& A )=0;
        virtual void prepareZeroAngleKeypoints(std::shared_ptr<FrameData> frame2)=0;
        void computeDescriptors(const cv::Mat& img, std::vector<cv::KeyPoint>& kpt, cv::Mat& desc);

        double calculateTotalLoss(cv::Mat& F_matrix,
                                    std::vector<std::shared_ptr<KeyPoint2>> matched_kpts1, 
                                    std::vector<std::shared_ptr<KeyPoint2>> matched_kpts2);

        bool validDescriptorRegion( double x, double y, int border );
        static int calculateDescriptorRadius(int patch_size, int kpt_size);

        // Test
        virtual cv::Mat collectDescriptorDistance(cv::Mat& y_k, std::shared_ptr<KeyPoint2> kpt2, cv::Mat& A, int reg_size=-1);
        
};

class DJETLoss : public LossFunction
{
    private:
        bool precompDescriptors;
        // int reg_size = n_reg_size;//7;

        cv::Mat img;
        std::vector<std::vector<cv::Mat>> descriptor_map;
    
    public:
        DJETLoss(cv::Mat& img, std::vector<std::shared_ptr<KeyPoint2>>& matched_kpts1, 
                                std::vector<std::shared_ptr<KeyPoint2>>& matched_kpts2);
        ~DJETLoss(){};

        bool doPrecomputeDescriptors()override;
        double calculateKptLoss(const cv::Mat& F_matrix, const cv::Mat& A_d_k, const cv::Mat& x_k, const cv::Mat& y_k, cv::Mat& v_k_opt)override;
        double calculateKptLoss(const cv::Mat& F_matrix, const std::shared_ptr<KeyPoint2> kpt1, const std::shared_ptr<KeyPoint2> kpt2, cv::Mat& v_k_opt)override;
        bool validKptLoc( double x, double y, int kpt_size )override;
        bool updateKeypoint( std::shared_ptr<KeyPoint2> kpt, double x_update, double y_update )override;
        void linearizeLossFunction( cv::Mat& y_k, std::shared_ptr<KeyPoint2> kpt2, cv::Mat& A )override;
        void computeDescriptors(const cv::Mat& img, std::vector<cv::KeyPoint>& kpt, cv::Mat& desc);

        cv::Mat collectDescriptorDistance( cv::Mat& y_k, std::shared_ptr<KeyPoint2> kpt2, cv::Mat& A, int reg_size=-1 )override;
        std::vector<cv::KeyPoint> generateLocalKpts( double kpt_x, double kpt_y, std::shared_ptr<KeyPoint2> kpt2, const cv::Mat& img, int reg_size_ );
        std::vector<cv::KeyPoint> generateLocalKptsNoAngle( double kpt_x, double kpt_y, std::shared_ptr<KeyPoint2> kpt2, const cv::Mat& img, int reg_size_ );
        //cv::Mat computeHammingDistance( cv::Mat& target_desc, cv::Mat& region_descs );
        void generateCoordinateVectors(double x_c, double y_c, int size, cv::Mat& x, cv::Mat& y);
        //void computeParaboloidNormalForAll( std::vector<std::shared_ptr<KeyPoint2>> matched_kpts1, std::vector<std::shared_ptr<KeyPoint2>> matched_kpts2, cv::Mat& img );
        
        void precomputeDescriptors( const cv::Mat& img );

        void printKptLoc( std::vector<cv::KeyPoint> kpts, int rows, int cols );
        void printLocalHammingDists( cv::Mat& hamming_dist_arr, int s );
        void printDescriptorMapFill();
        void printCalculatedDescsLog();

        void prepareZeroAngleKeypoints(std::shared_ptr<FrameData> frame2)override;
};

class ReprojectionLoss : public LossFunction
{
    public:
        ReprojectionLoss(cv::Mat& img);
        ~ReprojectionLoss(){};

        double calculateKptLoss( const cv::Mat& F_matrix, const cv::Mat& A_d_k, const cv::Mat& x_k, const cv::Mat& y_k, cv::Mat& v_k_opt)override;
        double calculateKptLoss(const cv::Mat& F_matrix, const std::shared_ptr<KeyPoint2> kpt1, const std::shared_ptr<KeyPoint2> kpt2, cv::Mat& v_k_opt)override;
        bool validKptLoc( double x, double y, int kpt_size )override;
        bool updateKeypoint( std::shared_ptr<KeyPoint2> kpt, double x_update, double y_update )override;
        void linearizeLossFunction( cv::Mat& y_k, std::shared_ptr<KeyPoint2> kpt2, cv::Mat& A )override;


        // Not used, just because of virtual function in base class.
        bool doPrecomputeDescriptors(){return false;};
        void prepareZeroAngleKeypoints(std::shared_ptr<FrameData> frame2){};

};


struct Point2DGJET
{
    Point2DGJET(double x, double y)
    {
        this->loc_[0] = x;
        this->loc_[1] = y;
    }
    double loc_[2];
};

class GJET : public PoseCalculator
{
    private:
        ParamID paramId = ParamID::LIEPARAM;

        // Settings
        bool linear;
        bool baseline;
        bool kpt_free;
        bool use_motion_prior;
        bool revert_kpt;
        int n_reg_size;
        double epsylon;

        // Toggle Functions
        bool iteration_log = true;
        bool match_score_loc = true;

        // Evaluation variables
        double avg_match_score = 0;
        double varianceN_match_score = 0;
        double avg_calculated_descs = 0;
        int n = 0;
        int n_matches = 0;

    public:
        GJET();
        ~GJET(){};

        int calculate( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2, cv::Mat& img )override;

        static double solveQuadraticFormForV( cv::Mat& A_k, cv::Mat& b_k, cv::Mat& c_k, cv::Mat& v_k );
        static cv::Mat solveKKT( cv::Mat& A, cv::Mat& g, cv::Mat& b, cv::Mat& h );
        static double epipolarConstrainedOptimization( const cv::Mat& F_matrix, const cv::Mat& A_d_k, const cv::Mat& x_k, const cv::Mat& y_k, cv::Mat& v_k_opt );
        void analysis(  std::shared_ptr<FrameData> frame1, 
                        std::shared_ptr<FrameData> frame2, 
                        cv::Mat& img )override;
        bool ceresLogToFile(int img_nr, ceres::Solver::Summary summary, std::string file_path="output/ceresLog.txt");
        bool logDescriptorDistance( int img_nr, 
                                    std::shared_ptr<LossFunction> loss_func,
                                    std::vector<std::shared_ptr<KeyPoint2>> m_kpts1,
                                    std::vector<std::shared_ptr<KeyPoint2>> m_kpts2);
};

class KeyPointUpdate : public ceres::EvaluationCallback
{
    private:
        bool updated = false;
        int it_num = 0;
        double step_size = 3.5; //px
        int outlier_threshold = 2; //px
        double* p;
        double best_loss = -1;
        cv::Mat img, K1, K2;
        std::shared_ptr<LossFunction> loss_func;
        std::shared_ptr<Parametrization> parametrization;
        std::vector<std::shared_ptr<KeyPoint2>> m_kpts1, m_kpts2;

        bool baseline;
        bool kpt_free;
    public:
        KeyPointUpdate(    cv::Mat& img, double* p, cv::Mat K1, cv::Mat K2, 
                            std::shared_ptr<LossFunction> loss_func, 
                            std::shared_ptr<Parametrization> parametrization);
        ~KeyPointUpdate(){};
        bool isUpdated();
        double getBestLoss();
        std::vector<std::shared_ptr<KeyPoint2>> getMKpts1();
        std::vector<std::shared_ptr<KeyPoint2>> getMKpts2();
        void logY_k_opt(std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2, cv::Mat F_matrix, std::vector<std::shared_ptr<Point2DGJET>> points2D);
        void registerOptKptPosReprErr( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2, cv::Mat& F_matrix );
        void registerOptKptPosLinear( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2);
        void PrepareForEvaluation(bool evaluate_jacobians, bool new_evaluation_point) final;
        double evaluate();

        bool updateKeypoint( std::shared_ptr<KeyPoint2> kpt, cv::Mat& v_k_opt );
        void revertKptsToInit();
        
        void addEvalKpt( std::shared_ptr<KeyPoint2> kpt1,
                         std::shared_ptr<KeyPoint2> kpt2);
        void setUpdated(bool value);

        double calculateScale(cv::Mat& v_k_opt);


        static void invalidateMatch(std::shared_ptr<KeyPoint2> kpt1, std::shared_ptr<KeyPoint2> kpt2);
        static void validateMatch(std::shared_ptr<KeyPoint2> kpt1, std::shared_ptr<KeyPoint2> kpt2);
        static bool validMatch(std::shared_ptr<KeyPoint2> kpt1, std::shared_ptr<KeyPoint2> kpt2);
        static void removeInvalidMatches(std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2);

        static void logOptLoc( std::shared_ptr<KeyPoint2> kpt );
        static void logKptState( std::shared_ptr<KeyPoint2> kpt, cv::Mat F_matrix );
};

class EarlyStoppingCheck : public ceres::IterationCallback
{
    private:
        KeyPointUpdate& itUpdate; 
    public:
        EarlyStoppingCheck(KeyPointUpdate& itUpdate) : itUpdate(itUpdate) {};
        ~EarlyStoppingCheck(){};

        ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary);
};

#endif