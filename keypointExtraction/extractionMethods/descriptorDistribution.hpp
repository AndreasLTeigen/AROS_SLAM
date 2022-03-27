#ifndef descDistribExtractor_h
#define descDistribExtractor_h

#include <string>
#include <opencv2/opencv.hpp>

#include "../keypointExtraction.hpp"
#include "../../dataStructures/keypoint.hpp"
#include "../../dataStructures/frameData.hpp"
#include "../../dataStructures/map3D.hpp"

class DescDistribExtractor : public Extractor
{
    private:
        int reg_size = 7;                                   // Size of local region of interest (around each keypoint)
        
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

        bool validDescriptorRegion( int x, int y, int W, int H, int border );
        std::vector<cv::KeyPoint> generateNeighbourhoodKpts( std::vector<cv::KeyPoint>& kpts, cv::Mat& img );
        void generateCoordinateVectors(double x_c, double y_c, int size, cv::Mat& x, cv::Mat& y);
        std::vector<cv::Mat> sortDescsN2( std::vector<cv::KeyPoint>& kpts, std::vector<cv::KeyPoint>& dummy_kpts, cv::Mat& desc, int reg_size );
        void sortDescsOrdered( cv::Mat& desc, std::vector<cv::Mat>& desc_ordered, int reg_size );
        void getCenterDesc( std::vector<cv::Mat>& desc_ordered, cv::Mat& desc_center );
        cv::Mat computeHammingDistance( cv::Mat& target_desc, cv::Mat& region_descs );

        cv::Mat computeHammingDistanceAnalysis( cv::KeyPoint target_kpt, cv::Mat& target_desc, std::vector<cv::KeyPoint> region_kpt, cv::Mat& region_descs );
        std::vector<cv::KeyPoint> generateDenseKeypoints(cv::Mat& img, float kpt_size=31);
        void testPrintKeypointOrdering(std::vector<cv::KeyPoint> dummy_kpts, int kpt_nr);
        void printSortKptsOrdered( std::vector<cv::KeyPoint>& kpts, int reg_size, int idx);
        void printLocalHammingDist( cv::Mat hamming_dists, int reg_size );
        void printLocalHammingDists( std::vector<cv::Mat> hamming_dists, int reg_size );
        cv::Mat generateKeypointCoverageMap(std::vector<cv::KeyPoint> kpts, int H, int W);

        void registerFrameKeypoints( std::shared_ptr<FrameData> frame, std::vector<cv::KeyPoint>& kpts, cv::Mat& desc, cv::Mat& center_desc, std::vector<cv::Mat>& A, std::vector<cv::Mat>& desc_hamming_dist );

    public:
        DescDistribExtractor(){};
        ~DescDistribExtractor(){};

        void extract( cv::Mat& img, std::shared_ptr<FrameData> frame, std::shared_ptr<Map3D> map_3d )override;
};

#endif