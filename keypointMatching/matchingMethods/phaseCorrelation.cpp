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
    double kpt2_x, kpt2_y;
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
        shift = cv::phaseCorrelate(desc1, desc2);

        center = kpt2->getDescriptor("center");
        kpt2_x = center.at<double>(0,0);
        kpt2_y = center.at<double>(1,0);
        kpt1->setCoordx(kpt2_x + shift.x);
        kpt1->setCoordy(kpt2_y + shift.y);
        

        // Registering the match.
        shared_ptr<Match> match = shared_ptr<Match>(new Match(kpt1, kpt2, 0, i));
        kpt1->addMatch(match, frame2->getFrameNr());
        kpt2->addMatch(match, frame1->getFrameNr());

        frame1->addKptToMatchList(kpt1, frame2);
        frame2->addKptToMatchList(kpt2, frame1);
    }
}