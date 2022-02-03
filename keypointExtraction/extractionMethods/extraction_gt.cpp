#include <memory>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "extraction_gt.hpp"
#include "bucketing.hpp"
#include "../../util/util.hpp"
#include "../../dataStructures/pose.hpp"
#include "../../dataStructures/mapPoint.hpp"

using std::vector;

void depthGTwBucketing( cv::Mat& img, std::shared_ptr<FrameData> frame, std::vector<cv::KeyPoint>& kpts, std::shared_ptr<Map3D> map_3d, int h_n_buckets, int w_n_buckets )
{   
    // Find <MapPoints> visible from current frame.
    bool pt_in_image_bounds;
    cv::Mat XYZ1;
    int kpt_id_it = 0;
    vector<std::shared_ptr<KeyPoint2>> reprojected_map_points;
    vector<std::shared_ptr<MapPoint>> map_points = map_3d->getAllMapPoints();

    for ( std::shared_ptr<MapPoint> map_point : map_points )
    {
        XYZ1 = map_point->getCoordXYZ();
        //std::cout << "XYZ1: " << XYZ1 << std::endl;
        cv::Mat xy1 = projectKpt( XYZ1 , frame->getGlobalPose(), frame->getKMatrix() );
        //std::cout << "R-Point: " << xy1 << std::endl;
        pt_in_image_bounds = isInsideImageBounds( xy1.at<double>(1,0), xy1.at<double>(0,0), img.size().height, img.size().width );
        if (pt_in_image_bounds)
        {
            std::shared_ptr<KeyPoint2> kpt = std::make_shared<KeyPoint2>( kpt_id_it, xy1, frame->getFrameNr() );
            kpt->setMapPoint(map_point);
            map_point->addObservation(kpt, map_point->getCoordXYZ(), cv::Mat::zeros(3, 1, CV_64F), frame->getGlobalPose());
            reprojected_map_points.push_back(kpt);
            kpt_id_it += 1;
        }
    }
    //std::cout << "Number of reprojected map points: " << reprojected_map_points.size() << std::endl;
    //std::cout << "Total number of map points: " << map_points.size() << std::endl;


    // Bucket keypoints found with detection algorithm in current image, naive approach.
    globalNaiveBucketing( img, kpts, h_n_buckets, w_n_buckets );

    // Convert new keypoints to <KeyPoint2>.
    vector<std::shared_ptr<KeyPoint2>> new_kpts;
    for ( cv::KeyPoint kpt : kpts)
    {
        new_kpts.push_back( std::make_shared<KeyPoint2>( kpt_id_it, kpt, frame->getFrameNr() ) );
        kpt_id_it += 1;
    }
    

    // Place one projected <MapPoints> into as many buckets (of size h * w) as possible.
    std::shared_ptr<KeyPoint2> dummy = std::make_shared<KeyPoint2>(-1, cv::Mat::zeros(3, 1, CV_64F), -1);
    vector<vector<vector<std::shared_ptr<KeyPoint2>>>> buckets ( h_n_buckets, vector<vector<std::shared_ptr<KeyPoint2>>> ( w_n_buckets, vector<std::shared_ptr<KeyPoint2>> (1, dummy) ) );

    int h_bucket, w_bucket;
    int bucket_height = img.size().height/h_n_buckets;
    int bucket_width = img.size().width/w_n_buckets;
    for ( std::shared_ptr<KeyPoint2> kpt : reprojected_map_points )
    {
        h_bucket = int(kpt->getCoordY() / bucket_height);
        w_bucket = int(kpt->getCoordX() / bucket_width);
        if ( buckets[h_bucket][w_bucket][0]->getKptId() == -1 )
        {
            buckets[h_bucket][w_bucket][0] = kpt;
        }
    }

    // If bucket is empty place keypoints detected in image in free buckets of size h * w.
    //std::cout << "New keypoints: " << new_kpts.size() << std::endl;
    for ( std::shared_ptr<KeyPoint2> kpt : new_kpts )
    {
        h_bucket = int(kpt->getCoordY() / bucket_height);
        w_bucket = int(kpt->getCoordX() / bucket_width);
        if ( buckets[h_bucket][w_bucket][0]->getKptId() == -1 )
        {
            buckets[h_bucket][w_bucket][0] = kpt;
        }
    }

    // Compiling final list of keypoints
    vector<std::shared_ptr<KeyPoint2>> reduced_kpts;
    for ( vector<vector<std::shared_ptr<KeyPoint2>>> w_buckets : buckets )
    {
        for ( vector<std::shared_ptr<KeyPoint2>> bucket : w_buckets )
        {
            if ( bucket[0]->getKptId() != -1 )
            {
                reduced_kpts.push_back(bucket[0]);
            }
        }
    }

    // Adding the keypoints in the buckets to current frame.
    //std::cout << "Remaining points: " << reduced_kpts.size() << std::endl;
    frame->registerKeypoints(reduced_kpts);
}