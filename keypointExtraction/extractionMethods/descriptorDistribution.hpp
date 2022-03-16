#ifndef descDistribExtractor_h
#define descDistribExtractor_h

#include <opencv2/opencv.hpp>

#include "../keypointExtraction.hpp"
#include "../../dataStructures/frameData.hpp"
#include "../../dataStructures/map3D.hpp"

class DescDistribExtractor : public Extractor
{
    private:
        int reg_size = 5;                                   // Size of local region of interest (around each keypoint)
        cv::Ptr<cv::ORB> orb = cv::ORB::create(500);
        //cv::Ptr<cv::ORB> descriptor = cv::ORB::create();

        std::vector<cv::KeyPoint> generateNeighbourhoodKpts( cv::KeyPoint kpt, int reg_size );
        std::vector<cv::KeyPoint> generateNeighbourhoodKpts( std::vector<cv::KeyPoint> kpts, int reg_size );
        std::vector<cv::KeyPoint> generateDenseKeypoints(cv::Mat& img, float kpt_size=31);
        std::vector<cv::Mat> sortDescs( std::vector<cv::KeyPoint>& kpts, std::vector<cv::KeyPoint>& dummy_kpts, cv::Mat& desc, int reg_size );
        cv::Mat computeHammingDistance( cv::Mat& target_desc, cv::Mat& region_descs, int N );

    public:
        DescDistribExtractor(){};
        ~DescDistribExtractor(){};

        void extract( cv::Mat& img, std::shared_ptr<FrameData> frame, std::shared_ptr<Map3D> map_3d )override;
};

#endif