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
        int edgeThreshold = 31;
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

        std::shared_ptr<Pose> calculate( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2 )override;
        void jointEpipolarOptimization( cv::Mat& F_matrix, std::vector<std::shared_ptr<KeyPoint2>>& matched_kpts1, std::vector<std::shared_ptr<KeyPoint2>>& matched_kpts2 );

        static double solveQuadraticFormForV( cv::Mat& A_k, cv::Mat& b_k, cv::Mat& c_k, cv::Mat& v_k );
        static cv::Mat solveKKT( cv::Mat& A, cv::Mat& g, cv::Mat& b, cv::Mat& h );
        static double epipolarConstrainedOptimization( const cv::Mat& F_matrix, const cv::Mat& A_d_k, const cv::Mat& x_k, const cv::Mat& y_k, cv::Mat& v_k_opt );


        void collectDescriptorDistances( cv::Mat& img, std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2 );
        std::vector<cv::KeyPoint> generateNeighbourhoodKpts( std::vector<cv::KeyPoint>& kpts, cv::Mat& img );
        void sortDescsOrdered( cv::Mat& desc, std::vector<cv::Mat>& desc_ordered, int reg_size );
        void getCenterDesc( std::vector<cv::Mat>& desc_ordered, cv::Mat& desc_center );
        cv::Mat computeHammingDistance( cv::Mat& target_desc, cv::Mat& region_descs );
        void generateCoordinateVectors(double x_c, double y_c, int size, cv::Mat& x, cv::Mat& y);
        void registerDDInfo( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2, cv::Mat& center_desc, std::vector<cv::Mat>& A );
        bool validDescriptorRegion( int x, int y, int W, int H, int border );
};

#endif