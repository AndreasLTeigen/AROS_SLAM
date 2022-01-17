#include <memory>
#include <opencv2/opencv.hpp>

#include "match.hpp"

using std::shared_ptr;
using std::weak_ptr;

Match::Match(shared_ptr<KeyPoint2> kpt1, 
             shared_ptr<KeyPoint2> kpt2, 
             float descr_distance,
             int match_id)
{
    this->kpt1 = kpt1;
    this->kpt2 = kpt2;
    this->setMatchID(match_id);
    this->setDescriptorDistance(descr_distance);
    this->setConfidence(1-descr_distance);
}

Match::~Match()
{
    //TODO: Implement destructor
    //std::cout << "MATCH DESTROYED " << this->getMatchID() << std::endl;
}


// Write functions
void Match::setMatchID(int match_id)
{
    std::unique_lock lock(this->mutex_match_id);
    this->match_id = match_id;
}

void Match::setConfidence(float confidence)
{
    std::unique_lock lock(this->mutex_confidence);
    this->confidence = confidence;
}

void Match::setDescriptorDistance(float descr_distance)
{
    std::unique_lock lock(this->mutex_descr_distance);
    this->descr_distance = descr_distance;
}


// Read functions
int Match::getMatchID()
{
    std::shared_lock(this->mutex_match_id);
    return this->match_id;
}

shared_ptr<KeyPoint2> Match::getKpt1()
{
    std::shared_lock(this->mutex_kpt1);
    return this->kpt1.lock();
}

shared_ptr<KeyPoint2> Match::getKpt2()
{
    std::shared_lock(this->mutex_kpt2);
    return this->kpt2.lock();
}

shared_ptr<KeyPoint2> Match::getConnectingKpt(int connecting_keypoint_frame_nr)
{
    std::shared_ptr<KeyPoint2> temp_kpt1 = this->getKpt1();
    std::shared_ptr<KeyPoint2> temp_kpt2 = this->getKpt2();

    if ( temp_kpt1->getObservationFrameNr() == connecting_keypoint_frame_nr )
    {
        return temp_kpt1;
    }
    else if ( temp_kpt2->getObservationFrameNr() == connecting_keypoint_frame_nr )
    {
        return temp_kpt2;
    }
    else
    {
        std::cout << "ERROR: <connecting_frame_nr> does not hold a frame connected to this pose" << std::endl;
        return nullptr;
    }
}

float Match::getConfidence()
{
    std::shared_lock(this->mutex_confidence);
    return this->confidence;
}

float Match::getDescrDistance()
{
    std::shared_lock(this->mutex_descr_distance);
    return this->descr_distance;
}