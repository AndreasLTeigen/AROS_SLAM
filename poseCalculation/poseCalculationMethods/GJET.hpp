#ifndef GJET_h
#define GJET_h

#include "../poseCalculation.hpp"
#include "../../dataStructures/frameData.hpp"
#include "../../dataStructures/keypoint.hpp"
#include "../../dataStructures/pose.hpp"
#include "../../dataStructures/parametrization.hpp"


class GJET : public PoseCalculator    // 5-Point with Outlier Rejection Pose Calculator
{
    private:
        int reg_size = 7;
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
                                                cv::ORB::HARRIS_SCORE,
                                                patchSize,
                                                fastThreshold);

    public:
        GJET(){};
        ~GJET(){};

        std::shared_ptr<Pose> calculate( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2, cv::Mat& img )override;
        void jointEpipolarOptimization( cv::Mat& F_matrix, std::vector<std::shared_ptr<KeyPoint2>>& matched_kpts1, std::vector<std::shared_ptr<KeyPoint2>>& matched_kpts2 );

        static double solveQuadraticFormForV( cv::Mat& A_k, cv::Mat& b_k, cv::Mat& c_k, cv::Mat& v_k );
        static cv::Mat solveKKT( cv::Mat& A, cv::Mat& g, cv::Mat& b, cv::Mat& h );
        static double epipolarConstrainedOptimization( const cv::Mat& F_matrix, const cv::Mat& A_d_k, const cv::Mat& x_k, const cv::Mat& y_k, cv::Mat& v_k_opt );

};

class DDNormal       // Descriptor distance Normalization
{
    private:
        int reg_size = 3;
        int step_size = 1; //px
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
                                                cv::ORB::HARRIS_SCORE,
                                                patchSize,
                                                fastThreshold);
    public:
        DDNormal(){};
        ~DDNormal(){};

        std::string orb_non_rot = "orb_non_rot";
        std::string quad_fit = "quad_fit";
        void collectDescriptorDistance( const cv::Mat& img, std::shared_ptr<KeyPoint2> kpt1, std::shared_ptr<KeyPoint2> kpt2 );
        void collectDescriptorDistances( cv::Mat& img, std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2 );
        std::vector<cv::KeyPoint> generateNeighbourhoodKpts( std::vector<std::shared_ptr<KeyPoint2>>& kpts, cv::Mat& img );
        void sortDescsOrdered( cv::Mat& desc, std::vector<cv::Mat>& desc_ordered, int reg_size );
        void getCenterDesc( std::vector<cv::Mat>& desc_ordered, cv::Mat& desc_center );
        cv::Mat computeHammingDistance( cv::Mat& target_desc, cv::Mat& region_descs );
        void generateCoordinateVectors(double x_c, double y_c, int size, cv::Mat& x, cv::Mat& y);
        void registerDDInfo( std::vector<std::shared_ptr<KeyPoint2>>& kpts, std::vector<cv::Mat>& A );
        bool validDescriptorRegion( double x, double y, int W, int H, int border );
        cv::Mat getNonRotatedDescriptors( cv::Mat& img, std::vector<cv::KeyPoint>& kpts );
        void registerNewDescriptors( std::vector<std::shared_ptr<KeyPoint2>> kpts, cv::Mat& desc);
        void registerNonRotDescs( std::vector<std::shared_ptr<KeyPoint2>> kpts, cv::Mat& img );
        void computeParaboloidNormalForAll( std::vector<std::shared_ptr<KeyPoint2>> matched_kpts1, std::vector<std::shared_ptr<KeyPoint2>> matched_kpts2, cv::Mat& img );

        void collectDescriptorDistancesNew( cv::Mat& img, std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2 );
        std::vector<cv::KeyPoint> generateLocalKpts( std::shared_ptr<KeyPoint2> kpt, const cv::Mat& img );
        void printKptLoc( std::vector<cv::KeyPoint> kpts, int rows, int cols );
        void printLocalHammingDists( cv::Mat& hamming_dist_arr, int s );

        bool updateKeypoint( std::shared_ptr<KeyPoint2> kpt, const cv::Mat& img );
        double calculateScale(cv::Mat& v_k_opt);
};


class DDNormalPrecomp   //Same as DDNormal but with precomputed descriptors.
{
        private:
            int reg_size = 3;
            int step_size = 1; //px
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
                                                    cv::ORB::HARRIS_SCORE,
                                                    patchSize,
                                                    fastThreshold);
        public:
            DDNormalPrecomp(){};
            ~DDNormalPrecomp(){};

            cv::Mat generateDescriptorMap( const cv::Mat& img );
};

#endif