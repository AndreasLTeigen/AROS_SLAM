#ifndef GJET_h
#define GJET_h

#include <ceres/ceres.h>

#include "../poseCalculation.hpp"
#include "../../dataStructures/frameData.hpp"
#include "../../dataStructures/keypoint.hpp"
#include "../../dataStructures/pose.hpp"
#include "../../dataStructures/parametrization.hpp"


class GJET : public PoseCalculator
{
    private:
        ParamID paramId = ParamID::STDPARAM;

    public:
        GJET(){};
        ~GJET(){};

        std::shared_ptr<Pose> calculate( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2, cv::Mat& img )override;
        void jointEpipolarOptimization( cv::Mat& F_matrix, std::vector<std::shared_ptr<KeyPoint2>>& matched_kpts1, std::vector<std::shared_ptr<KeyPoint2>>& matched_kpts2 );

        static double solveQuadraticFormForV( cv::Mat& A_k, cv::Mat& b_k, cv::Mat& c_k, cv::Mat& v_k );
        static cv::Mat solveKKT( cv::Mat& A, cv::Mat& g, cv::Mat& b, cv::Mat& h );
        static double epipolarConstrainedOptimization( const cv::Mat& F_matrix, const cv::Mat& A_d_k, const cv::Mat& x_k, const cv::Mat& y_k, cv::Mat& v_k_opt );

        static double reprojectionError(const cv::Mat& F_matrix, const cv::Mat& x_k, const cv::Mat& y_k, cv::Mat& v_k_opt);

};

class LossFunction
{
    protected:
        int W, H;
        ParamID paramId = ParamID::STDPARAM;

        int nfeatures = 500;
        float scaleFactor = 1.2f;
        int nlevels = 8;
        int edgeThreshold = 19;
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
    public:
        LossFunction(cv::Mat& img);
        LossFunction(int W, int H);
        ~LossFunction(){};

        // TODO: Move these variables to a new place
        std::string descriptor_name = "orb";
        int getPatchSize();

        virtual double calculateKptLoss(const cv::Mat& F_matrix, const cv::Mat& A_d_k, const cv::Mat& x_k, const cv::Mat& y_k, cv::Mat& v_k_opt)=0;
        virtual double calculateKptLoss(const cv::Mat& F_matrix, const std::shared_ptr<KeyPoint2> kpt1, const std::shared_ptr<KeyPoint2> kpt2, cv::Mat& v_k_opt)=0;
        virtual bool validKptLoc( double x, double y, int kpt_size )=0;
        virtual bool updateKeypoint( std::shared_ptr<KeyPoint2> kpt, const cv::Mat& img, double x_update, double y_update )=0;
        virtual void linearizeLossFunction(cv::Mat& img, std::shared_ptr<KeyPoint2> kpt1, std::shared_ptr<KeyPoint2> kpt2 )=0;
        void computeDescriptors(const cv::Mat& img, std::vector<cv::KeyPoint>& kpt, cv::Mat& desc);

        bool validDescriptorRegion( double x, double y, int border );
        static int calculateDescriptorRadius(int patch_size, int kpt_size);
        
};

class DJETLoss : public LossFunction
{
    private:
        bool precompDescriptors = true;
        int reg_size = 9;
        std::vector<std::vector<cv::Mat>> descriptor_map;
    public:
        DJETLoss(cv::Mat& img, std::vector<std::shared_ptr<KeyPoint2>>& matched_kpts1, 
                                std::vector<std::shared_ptr<KeyPoint2>>& matched_kpts2);
        ~DJETLoss(){};

        double calculateKptLoss(const cv::Mat& F_matrix, const cv::Mat& A_d_k, const cv::Mat& x_k, const cv::Mat& y_k, cv::Mat& v_k_opt)override;
        double calculateKptLoss(const cv::Mat& F_matrix, const std::shared_ptr<KeyPoint2> kpt1, const std::shared_ptr<KeyPoint2> kpt2, cv::Mat& v_k_opt)override;
        bool validKptLoc( double x, double y, int kpt_size )override;
        bool updateKeypoint( std::shared_ptr<KeyPoint2> kpt, const cv::Mat& img, double x_update, double y_update )override;
        void linearizeLossFunction(cv::Mat& img, std::shared_ptr<KeyPoint2> kpt1, std::shared_ptr<KeyPoint2> kpt2 )override;
        void computeDescriptors(const cv::Mat& img, std::vector<cv::KeyPoint>& kpt, cv::Mat& desc);

        void collectDescriptorDistance( const cv::Mat& img, std::shared_ptr<KeyPoint2> kpt1, std::shared_ptr<KeyPoint2> kpt2 );
        std::vector<cv::KeyPoint> generateLocalKpts( std::shared_ptr<KeyPoint2> kpt, const cv::Mat& img );
        cv::Mat computeHammingDistance( cv::Mat& target_desc, cv::Mat& region_descs );
        void generateCoordinateVectors(double x_c, double y_c, int size, cv::Mat& x, cv::Mat& y);
        void computeParaboloidNormalForAll( std::vector<std::shared_ptr<KeyPoint2>> matched_kpts1, std::vector<std::shared_ptr<KeyPoint2>> matched_kpts2, cv::Mat& img );
        
        void precomputeDescriptors( const cv::Mat& img );

        void printKptLoc( std::vector<cv::KeyPoint> kpts, int rows, int cols );
        void printLocalHammingDists( cv::Mat& hamming_dist_arr, int s );
        void printDescriptorMapFill();
};

class ReprojectionLoss : public LossFunction
{
    public:
        ReprojectionLoss(cv::Mat& img);
        ~ReprojectionLoss(){};

        double calculateKptLoss( const cv::Mat& F_matrix, const cv::Mat& A_d_k, const cv::Mat& x_k, const cv::Mat& y_k, cv::Mat& v_k_opt)override;
        double calculateKptLoss(const cv::Mat& F_matrix, const std::shared_ptr<KeyPoint2> kpt1, const std::shared_ptr<KeyPoint2> kpt2, cv::Mat& v_k_opt)override;
        bool validKptLoc( double x, double y, int kpt_size )override;
        bool updateKeypoint( std::shared_ptr<KeyPoint2> kpt, const cv::Mat& img, double x_update, double y_update )override;
        void linearizeLossFunction( cv::Mat& img, std::shared_ptr<KeyPoint2> kpt1, std::shared_ptr<KeyPoint2> kpt2 )override;
};


class KeyPointUpdate : public ceres::EvaluationCallback
{
    private:
        int step_size = 1; //px
        double* p;
        cv::Mat img, K1, K2;
        std::shared_ptr<LossFunction> loss_func;
        std::shared_ptr<Parametrization> parametrization;
        std::vector<std::shared_ptr<KeyPoint2>> m_kpts1, m_kpts2;
    public:
        KeyPointUpdate(    cv::Mat& img, double* p, cv::Mat K1, cv::Mat K2, 
                            std::shared_ptr<LossFunction> loss_func, 
                            std::shared_ptr<Parametrization> parametrization);
        ~KeyPointUpdate(){};
        
        void PrepareForEvaluation(bool evaluate_jacobians, bool new_evaluation_point) final;

        bool updateKeypoint( std::shared_ptr<KeyPoint2> kpt, const cv::Mat& img );
        void moveKptsToOptLoc(const cv::Mat& img);

        void addEvalKpt( std::shared_ptr<KeyPoint2> kpt1,
                         std::shared_ptr<KeyPoint2> kpt2);

        double calculateScale(cv::Mat& v_k_opt);


        static void invalidateMatch(std::shared_ptr<KeyPoint2> kpt1, std::shared_ptr<KeyPoint2> kpt2);
        static bool validMatch(std::shared_ptr<KeyPoint2> kpt1, std::shared_ptr<KeyPoint2> kpt2);

        static void logOptLoc( std::shared_ptr<KeyPoint2> kpt );
        static void logKptState( std::shared_ptr<KeyPoint2> kpt, cv::Mat F_matrix );
};

#endif