#include <vector>
#include <opencv2/opencv.hpp>

#include "bruteForceMatching.hpp"
#include "../../dataStructures/frameData.hpp"

using cv::Mat;
using cv::DMatch;
using std::vector;
using std::shared_ptr;

BFMatcher::BFMatcher(const YAML::Node config)
{
    std::cout << std::left;
    std::cout << std::setw(20) << "Matcher:" << "Brute force" << std::endl;

    this->do_lowes_ratio_test = config["do_lowes_ratio_test"].as<bool>();
    this->retain_N_best_matches = config["retain_N_best_matches"].as<int>();
}

int BFMatcher::matchKeypoints( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2 )
{
    if (this->two_way_matching)
    {
        return this->bfMatchTwoWay(frame1, frame2);
    }
    else
    {
        return this->bfMatch(frame1, frame2);
    }
}

int BFMatcher::bfMatch(std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2)
{
    //--Performing brute force matching without cross check and normalized hamming distance
    
    vector<vector<DMatch>> matches, matches_temp;
    Mat queryDesc = frame1->compileCVDescriptors();
    Mat trainDesc = frame2->compileCVDescriptors();

    this->matcher.knnMatch(queryDesc, trainDesc, matches_temp, 2);

    //--Lowes ratio test
    if (this->do_lowes_ratio_test)
    {
        for( int i = 0; i < matches_temp.size(); ++i)
        {
            if (matches_temp[i][0].distance < 0.8*matches_temp[i][1].distance)
            {
                vector<DMatch> temp;
                temp.push_back(matches_temp[i][0]);
                matches.push_back(temp);
            }
        }
    }
    else
    {
        for ( int i = 0; i < matches_temp.size(); ++i )
        {
            vector<DMatch> temp;
            temp.push_back(matches_temp[i][0]);
            matches.push_back(temp);
        }
    }

    FrameData::registerMatches(frame1, frame2, matches);

    int num_matches = frame1->getMatchedKeypoints(frame2->getFrameNr()).size();

    this->num_matches = num_matches;

    if (num_matches == 0)
    {
        std::cerr << "ERROR(frame " << frame1->getImgId() << "): No matches found" << std::endl;
        return 1;
    }

    if (this->retain_N_best_matches != -1)
    {
        this->matchPruning(frame1, frame2, this->retain_N_best_matches);
    }

    return 0;
}

int BFMatcher::bfMatchTwoWay(std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2)
{
    //--Performing brute force matching with cross check
    
    vector<vector<DMatch>> matches, matches_temp;
    Mat queryDesc = frame1->compileCVDescriptors();
    Mat trainDesc = frame2->compileCVDescriptors();

    this->matcher.knnMatch(queryDesc, trainDesc, matches_temp, 2);

    //--Lowes ratio test
    if (this->do_lowes_ratio_test)
    {
        for( int i = 0; i < matches_temp.size(); ++i)
        {
            if (matches_temp[i][0].distance < 0.8*matches_temp[i][1].distance)
            {
                vector<DMatch> temp;
                temp.push_back(matches_temp[i][0]);
                matches.push_back(temp);
            }
        }
    }
    else
    {
        for ( int i = 0; i < matches_temp.size(); ++i )
        {
            vector<DMatch> temp;
            temp.push_back(matches_temp[i][0]);
            matches.push_back(temp);
        }
    }

    // Two way matching.
    int N = matches.size();
    vector<uint16_t> match_idxs(frame2->getNumKeypoints(), 
                                std::numeric_limits<uint16_t>::max());
    int match_idx, kpt2_idx;
    int match_cnt = 0;
    for (int i = 0; i < N; ++i)
    {
        kpt2_idx = matches[i][0].trainIdx;
        if(match_idxs[kpt2_idx] == std::numeric_limits<uint16_t>::max())
        {
            match_idxs[kpt2_idx] = i;
            match_cnt += 1;
        }
        else
        {
            match_idx = match_idxs[kpt2_idx];
            if(matches[match_idx][0].distance > matches[i][0].distance)
            {
                match_idxs[kpt2_idx] = i;
            }
        }
    }

    vector<vector<cv::DMatch>> matches_two_way(match_cnt);

    uint16_t it = 0;
    for (int i = 0; i < match_idxs.size(); ++i)
    {
        if (match_idxs[i] != std::numeric_limits<uint16_t>::max())
        {
            matches_two_way[it] = matches[match_idxs[i]];
            it += 1;
        }
    }

    FrameData::registerMatches(frame1, frame2, matches_two_way);

    int num_matches = frame1->getMatchedKeypoints(frame2->getFrameNr()).size();

    this->num_matches = num_matches;

    if (num_matches == 0)
    {
        std::cerr << "ERROR(frame " << frame1->getImgId() << "): No matches found" << std::endl;
        return 1;
    }


    if (this->retain_N_best_matches != -1)
    {
        this->matchPruning(frame1, frame2, this->retain_N_best_matches);
    }

    return 0;
}

void BFMatcher::matchPruning(   shared_ptr<FrameData> frame1,
                                shared_ptr<FrameData> frame2,
                                int N_remaining)
{
    /*
    Prunes all but the <N_remaining> matches. Matches are pruned based on
    their Hamming distance.

    Arguments:
        frameX:         Frames whose matches with eachother will be pruned.
        N_remaining:    Number of remaining matches after pruning.
    
    Note:
        <frameX->matched_keypoints> will be directly updated.
    */

    vector<shared_ptr<KeyPoint2>> match_kpts1 = frame1->getMatchedKeypoints(frame2->getFrameNr());
    vector<shared_ptr<KeyPoint2>> match_kpts2 = frame2->getMatchedKeypoints(frame1->getFrameNr());
    vector<int> idxs(N_remaining,-1);
    vector<double> dists(N_remaining, 255.0);

    for (int i = 0; i < match_kpts1.size(); ++i)
    {
        shared_ptr<KeyPoint2> match_kpt1 = match_kpts1[i];
        double desc_dist = match_kpt1->getHighestConfidenceMatch(frame2->getFrameNr())->getDescrDistance();

        std::vector<double>::iterator it = std::max_element(std::begin(dists), std::end(dists));
        int largest_retained_dist_idx = std::distance(std::begin(dists), it);

        if (dists[largest_retained_dist_idx] > desc_dist)
        {
            idxs[largest_retained_dist_idx] = i;
            dists[largest_retained_dist_idx] = desc_dist;
        }
    }

    cv::Mat inliers = cv::Mat::zeros(1, match_kpts1.size(), CV_8U);
    for (int i = 0; i < match_kpts1.size(); ++i)
    {
        for (int j = 0; j < idxs.size(); ++j)
        {
            if (idxs[j] == i)
            {
                inliers.at<uchar>(i) = 255;
            }
        }
    }
    FrameData::removeOutlierMatches(inliers, frame1, frame2);
}