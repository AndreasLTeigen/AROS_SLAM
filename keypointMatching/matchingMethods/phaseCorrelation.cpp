#include <opencv2/opencv.hpp>

#include "phaseCorrelation.hpp"
#include "../../dataStructures/keypoint.hpp"
#include "../../dataStructures/frameData.hpp"
#include "../../dataStructures/match.hpp"
#include "../../util/util.hpp"

using std::vector;
using std::shared_ptr;

void PhaseCorrelation::matchKeypoints( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2 )
{   
    double kpt1_x, kpt1_y;
    cv::Point2d shift;
    cv::Mat desc1, desc2, center;
    shared_ptr<KeyPoint2> kpt1, kpt2;
    vector<shared_ptr<KeyPoint2>> kpts1, kpts2;
    
    kpts1 = frame1->getKeypoints();
    kpts2 = frame2->getKeypoints();

    // Assuming grid for both frames are of equal size and that <kpts1.size()> = <kpts2.size()>.
    for ( int i = 0; i < kpts1.size(); ++i )
    {
        // Finding block shift and setting the correct keypoint location.
        kpt1 = kpts1[i];
        kpt2 = kpts2[i];
        desc1 = kpt1->getDescriptor("block_feature");
        desc2 = kpt2->getDescriptor("block_feature");
        desc1.convertTo(desc1, CV_32FC1, 1.0/255.0);
        desc2.convertTo(desc2, CV_32FC1, 1.0/255.0);

        shift = cv::phaseCorrelate(desc2, desc1);

        /*
        cv::imshow("1", desc1);
        cv::imshow("2", desc2);
        std::cout << kpt1->getDescriptor("center") << std::endl;
        cv::waitKey(0);
        
        std::cout << shift << std::endl;
        */

        if ( cv::norm(shift) >= this->shift_threshold )
        {
            center = kpt1->getDescriptor("center");
            kpt1_x = center.at<double>(0,0);
            kpt1_y = center.at<double>(1,0);
            kpt2->setCoordx(kpt1_x - shift.x);
            kpt2->setCoordy(kpt1_y - shift.y);
            

            // Registering the match.
            shared_ptr<Match> match = shared_ptr<Match>(new Match(kpt1, kpt2, 0, i));
            kpt1->addMatch(match, frame2->getFrameNr());
            kpt2->addMatch(match, frame1->getFrameNr());

            frame1->addKptToMatchList(kpt1, frame2);
            frame2->addKptToMatchList(kpt2, frame1);
        }
    }
}

void KLTTracker::matchKeypoints( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2 )
{   
    double kpt1_x, kpt1_y;
    vector<cv::Point2f> pts1, pts2;
    cv::Mat center, error, status, img1;
    shared_ptr<KeyPoint2> kpt1, kpt2;
    vector<shared_ptr<KeyPoint2>> kpts1, kpts2;

    img1 = frame1->getImg();
    this->winSize = cv::Size(img1.cols, img1.rows);
    
    kpts1 = frame1->getKeypoints();
    kpts2 = frame2->getKeypoints();

    for ( int i = 0; i < kpts1.size(); ++i )
    {
        center = kpts1[i]->getDescriptor("center");
        pts1.push_back(cv::Point2f(center.at<double>(0,0), center.at<double>(1,0)));
    }


    cv::calcOpticalFlowPyrLK(frame1->getImg(), frame2->getImg(), pts1, pts2, status, error);

    for ( int i = 0; i < pts1.size(); ++i )
    {
        kpt1 = kpts1[i];
        kpt2 = kpts2[i];
        kpt2->setCoordx(pts2[i].x);
        kpt2->setCoordy(pts2[i].y);

        if (cv::norm(pts2[i]-pts1[i]) >= this->shift_threshold)
        {
            // Registering the match.
            shared_ptr<Match> match = shared_ptr<Match>(new Match(kpt1, kpt2, 0, i));
            kpt1->addMatch(match, frame2->getFrameNr());
            kpt2->addMatch(match, frame1->getFrameNr());

            frame1->addKptToMatchList(kpt1, frame2);
            frame2->addKptToMatchList(kpt2, frame1);
        }
    }
}
/*
    for ( int i = 0; i < kpts1.size(); ++i )
    {
        center = kpts2[i]->getDescriptor("center");
        //pts2.push_back(cv::Point(center.at<double>(0,0), center.at<double>(1,0)));
        loc = (cv::Mat_<double>(2,1) << center.at<double>(0,0), center.at<double>(1,0));
        pts2.push_back(loc);
    }


    cv::calcOpticalFlowPyrLK(frame2->getImg(), frame1->getImg(), pts2, pts1, status, error);

    for ( int i = 0; i < pts1.rows; ++i )
    {
        kpts1[i]->setCoordx(pts1.at<double>(i,0));
        kpts1[i]->setCoordy(pts1.at<double>(i,1));
    }

    */