#include <map>
#include <math.h>
#include <memory>
#include <vector>
#include <shared_mutex>
#include <opencv2/opencv.hpp>

#include "keypoint.hpp"
#include "frameData.hpp"

using std::map;
using std::vector;
using std::shared_ptr;
using std::unique_ptr;
using cv::Mat;


KeyPoint2::KeyPoint2( int observation_frame_nr, int kpt_id, cv::KeyPoint kpt, Mat descr, std::string descr_type )
{
    this->setKptID(kpt_id);
    this->setCoordx(kpt.pt.x);
    this->setCoordy(kpt.pt.y);
    this->setSize(kpt.size);
    this->setAngle(kpt.angle);
    this->setResponse(kpt.response);
    this->setOctave(kpt.octave);
    this->setObservationFrameNr(observation_frame_nr);
    this->setDescriptor(descr, descr_type);
    this->setMapPoint(nullptr);
}

KeyPoint2::~KeyPoint2()
{
    //TODO: Implement destructor
    //std::cout << "Keypoint Destroyed" << std::endl;
}

void KeyPoint2::setKptID(int kpt_id)
{
    std::unique_lock lock(this->mutex_kpt_id);
    this->kpt_id = kpt_id;
}

void KeyPoint2::setCoordx(double x)
{
    std::unique_lock lock(this->mutex_coord);
    this->x = x;
}

void KeyPoint2::setCoordy(double y)
{
    std::unique_lock lock(this->mutex_coord);
    this->y = y;
}

void KeyPoint2::setAngle(double angle)
{
    std::unique_lock lock(this->mutex_angle);
    this->angle = angle;
}

void KeyPoint2::setOctave(int octave)
{
    std::unique_lock lock(this->mutex_octave);
    this->octave = octave;
}

void KeyPoint2::setResponse(int response)
{
    std::unique_lock lock(this->mutex_response);
    this->response = response;
}

void KeyPoint2::setSize(int size)
{
    std::unique_lock lock(this->mutex_size);
    this->size = size;
}

void KeyPoint2::setObservationFrameNr(int observation_frame_nr)
{
    std::unique_lock lock(this->mutex_observation_frame_nr);
    this->observation_frame_nr = observation_frame_nr;
}

void KeyPoint2::setDescriptor(Mat descr, std::string descr_type)
{
    std::unique_lock lock(this->mutex_descriptors);
    this->descriptors[descr_type] = descr;
}

void KeyPoint2::setMapPoint(std::shared_ptr<MapPoint> map_point)
{
    std::unique_lock lock(this->mutex_map_point);
    this->map_point = map_point;
}

void KeyPoint2::addMatch(shared_ptr<Match> match, int matched_frame_nr)
{
    std::unique_lock lock(this->mutex_matches_map);
    this->matches[matched_frame_nr].push_back(match);
}

void KeyPoint2::removeAllMatchReferences(int matched_frame_nr)
{
    // Removes references to all keypoint's matches with <matched_frame_nr>
    // WARNING: DOES NOT REMOVE THE REFERENCES TO THE MATCH FOR THE MATCHED KEYPOINT
    std::unique_lock lock(this->mutex_matches_map);

    this->matches.erase( matched_frame_nr ); 
}

void KeyPoint2::removeAllMatches(int matched_frame_nr)
{
    std::shared_ptr<KeyPoint2> temp_kpt;

    std::unique_lock lock(this->mutex_matches_map);
    for ( shared_ptr<Match> match : this->matches[matched_frame_nr] )
    {
        temp_kpt = match->getConnectingKpt(matched_frame_nr);
        temp_kpt->removeAllMatchReferences(this->getObservationFrameNr());
    }
    this->matches.erase( matched_frame_nr );
}

void KeyPoint2::orderMatchesByConfidence(int matched_frame_nr)
{
    //TODO: MAKE THIS FUNCTION, SOME FUNCTION RELIES ON AN ORDERED KEYPOINT LIST
}

int KeyPoint2::getKptId()
{
    std::shared_lock lock(this->mutex_kpt_id);
    return this->kpt_id;
}

double KeyPoint2::getCoordX()
{
    std::shared_lock lock(this->mutex_coord);
    return this->x;
}

double KeyPoint2::getCoordY()
{
    std::shared_lock lock(this->mutex_coord);
    return this->y;
}

int KeyPoint2::getOctave()
{
    std::shared_lock lock(this->mutex_octave);
    return this->octave;
}

int KeyPoint2::getObservationFrameNr()
{
    std::shared_lock lock(this->mutex_observation_frame_nr);
    return this->observation_frame_nr;
}

double KeyPoint2::getAngle()
{
    std::shared_lock lock(this->mutex_angle);
    return this->angle;
}

double KeyPoint2::getResponse()
{
    std::shared_lock lock(this->mutex_response);
    return this->response;
}

double KeyPoint2::getSize()
{
    std::shared_lock lock(this->mutex_size);
    return this->size;
}

Mat KeyPoint2::getDescriptor(std::string descr_type)
{
    std::shared_lock lock(mutex_descriptors);
    return this->descriptors[descr_type];
}

std::shared_ptr<MapPoint> KeyPoint2::getMapPoint()
{
    std::shared_lock lock(this->mutex_map_point);
    return this->map_point;
}

vector<shared_ptr<Match>> KeyPoint2::getMatches(int matched_frame_nr)
{
    std::shared_lock lock(this->mutex_matches_map);
    return this->matches[matched_frame_nr];
}

shared_ptr<Match> KeyPoint2::getHighestConfidenceMatch(int matched_frame_nr)
{
    // Assumes a sorted list
    if (this->getMatches(matched_frame_nr).size() == 0)
    {
        return nullptr;
    }
    else
    {
        return this->getMatches(matched_frame_nr)[0];
    }
}

cv::Point KeyPoint2::compileCV2DPoint()
{
    unique_ptr<cv::Point> point = unique_ptr<cv::Point>(new cv::Point(this->getCoordX(), this->getCoordY()));
    return *point;
}

Mat KeyPoint2::compileHomogeneousCV2DPoint()
{
    return (cv::Mat_<double>(3,1) << this->getCoordX(), this->getCoordY(), 1);
}

// Static functions
double KeyPoint2::calculateKeypointDistance(shared_ptr<KeyPoint2> kpt1, shared_ptr<KeyPoint2> kpt2)
{
    return sqrt((kpt1->getCoordX() - kpt2->getCoordX())*(kpt1->getCoordX() - kpt2->getCoordX()) + (kpt1->getCoordY() - kpt2->getCoordY())*(kpt1->getCoordY() - kpt2->getCoordY()));
}