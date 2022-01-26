#include <memory>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "bucketing.hpp"
#include "../../dataStructures/frameData.hpp"
#include "../../dataStructures/keypoint.hpp"

using std::vector;


// ############################ GLOBAL NAIVE BUCKETING FRAME BASED ############################

void globalNaiveBucketing(cv::Mat& img, std::shared_ptr<FrameData> frame, int h_n_buckets, int w_n_buckets)
{
    /*
    Arguments:
        img:            Input image the keypoints have been colected for
        frame:          Current frame containing all keypoints.
        h_n_buckets:    Number of buckets counting vertically
        w_n_buckets:    Number of buckets counting horisontally
        n:              Max number of keypoints remaining in each bucket
    */
    // Keypoint agnostic, aka. keypoints need to be calculated beforehand
    // Selects the best 'n' keypoints in every 'bucket'

    cv::Size s = img.size();
    int h_img = s.height;
    int w_img = s.width;
    int bucket_height = h_img/h_n_buckets;
    int bucket_width = w_img/w_n_buckets;

    int n_buckets = h_n_buckets*w_n_buckets;
    
    int h_bucket, w_bucket;

    // Initializing 2D matrix of buckets
    vector<vector<vector<std::shared_ptr<KeyPoint2>>>> buckets ( h_n_buckets, vector<vector<std::shared_ptr<KeyPoint2>>> ( w_n_buckets, vector<std::shared_ptr<KeyPoint2>> (1, nullptr)) );

    for ( std::shared_ptr<KeyPoint2> kpt : frame->getKeypoints() )
    {
        h_bucket = int(kpt->getCoordY() / bucket_height);
        w_bucket = int(kpt->getCoordX() / bucket_width);
        buckets[h_bucket][w_bucket].push_back(kpt);
    }

    print2DBuckets(buckets);
    std::cout << "Keypoints len 1: " << frame->getKeypoints().size() << std::endl;

    std::shared_ptr<KeyPoint2> kpt;
    vector<std::shared_ptr<KeyPoint2>> best_keypoints;
    for (vector<vector<std::shared_ptr<KeyPoint2>>> w_buckets : buckets)
    {
        for (vector<std::shared_ptr<KeyPoint2>> bucket : w_buckets)
        {
            kpt = getBestKeypointInBucket(bucket);
            if (kpt != nullptr)
            {
                best_keypoints.push_back(kpt);
            }
        }
    }
    std::cout << "Keypoints len 2: " << best_keypoints.size() << std::endl;

    frame->setAllKeypoints(best_keypoints);

    std::cout << "Keypoints len 3: " << frame->getKeypoints().size() << std::endl;
}

void print2DBuckets(vector<vector<vector<std::shared_ptr<KeyPoint2>>>> buckets)
{
    std::cout << "Bucket len: " << std::endl;
    for (vector<vector<std::shared_ptr<KeyPoint2>>> w_buckets : buckets)
    {
        for (vector<std::shared_ptr<KeyPoint2>> bucket : w_buckets)
        {
            std::cout << bucket.size() << " ";
        }
        std::cout << "\n";
    }
}

std::shared_ptr<KeyPoint2> getBestKeypointInBucket(vector<std::shared_ptr<KeyPoint2>> bucket)
{

    std::shared_ptr<KeyPoint2> best_kpt = bucket[0];
    
    for ( std::shared_ptr<KeyPoint2> kpt : bucket )
    {
        if (best_kpt == nullptr && kpt != nullptr)
        {
            best_kpt = kpt;
        }
        else if (best_kpt == nullptr && kpt == nullptr)
        {
            continue;
        }
        else if (kpt->getResponse() > best_kpt->getResponse())
        {
            best_kpt = kpt;
        }
    }
    return best_kpt;
}



// #############################################################################################





// ############################ GLOBAL NAIVE BUCKETING OPENCV BASED ############################

void globalNaiveBucketing(cv::Mat& img, std::vector<cv::KeyPoint>& kpts, int h_n_buckets, int w_n_buckets)
{
    /*
    Arguments:
        img:            Input image the keypoints have been colected for
        kpts:           List of all keypoints.
        h_n_buckets:    Number of buckets counting vertically
        w_n_buckets:    Number of buckets counting horisontally
        n:              Max number of keypoints remaining in each bucket
    Returns:
        kpts:           Variable changed to reduced form after bucketing.
    */
    // Keypoint agnostic, aka. keypoints need to be calculated beforehand
    // Selects the best 'n' keypoints in every 'bucket'

    cv::Size s = img.size();
    int h_img = s.height;
    int w_img = s.width;
    int bucket_height = h_img/h_n_buckets;
    int bucket_width = w_img/w_n_buckets;

    int n_buckets = h_n_buckets*w_n_buckets;
    
    int h_bucket, w_bucket;

    // Initializing 2D matrix of buckets
    cv::KeyPoint dummy = cv::KeyPoint(-1,-1, 1);
    vector<vector<vector<cv::KeyPoint>>> buckets ( h_n_buckets, vector<vector<cv::KeyPoint>> ( w_n_buckets, vector<cv::KeyPoint> (1, dummy) ) );

    for ( cv::KeyPoint kpt : kpts )
    {
        h_bucket = int(kpt.pt.y / bucket_height);
        w_bucket = int(kpt.pt.x / bucket_width);
        buckets[h_bucket][w_bucket].push_back(kpt);
    }

    print2DBuckets(buckets);
    std::cout << "Keypoints len 1: " << kpts.size() << std::endl;

    cv::KeyPoint kpt;
    vector<cv::KeyPoint> best_keypoints;
    for (vector<vector<cv::KeyPoint>> w_buckets : buckets)
    {
        for (vector<cv::KeyPoint> bucket : w_buckets)
        {
            kpt = getBestKeypointInBucket(bucket);
            if (kpt.pt.y != -1)
            {
                best_keypoints.push_back(kpt);
            }
        }
    }
    std::cout << "Keypoints len 2: " << best_keypoints.size() << std::endl;

    kpts = best_keypoints;
}

void print2DBuckets(vector<vector<vector<cv::KeyPoint>>> buckets)
{
    std::cout << "Bucket len: " << std::endl;
    for (vector<vector<cv::KeyPoint>> w_buckets : buckets)
    {
        for (vector<cv::KeyPoint> bucket : w_buckets)
        {
            std::cout << bucket.size() << " ";
        }
        std::cout << "\n";
    }
}

cv::KeyPoint getBestKeypointInBucket(vector<cv::KeyPoint> bucket)
{

    cv::KeyPoint best_kpt = bucket[0];
    
    for ( cv::KeyPoint kpt : bucket )
    {
        if (best_kpt.pt.y == -1 && kpt.pt.y != -1)
        {
            best_kpt = kpt;
        }
        else if (best_kpt.pt.y == -1 && kpt.pt.y == -1)
        {
            continue;
        }
        else if (kpt.response > best_kpt.response)
        {
            best_kpt = kpt;
        }
    }
    return best_kpt;
}



// ###############################################################################