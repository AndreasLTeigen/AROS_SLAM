#include <memory>
#include <opencv2/opencv.hpp>

#include "mapPoint.hpp"
#include "keypoint.hpp"


using cv::Mat;
using std::map;
using std::weak_ptr;
using std::shared_ptr;

MapPoint::MapPoint(int id, double X, double Y, double Z, double std_X, double std_Y, double std_Z, shared_ptr<KeyPoint2> kpt1, Mat T1)
{
    // Constructor for map point with depth prior
    this->id = id;
    this->observation_cnt = 1;
    this->setCoordX(X);
    this->setCoordY(Y);
    this->setCoordZ(Z);
    this->setSTDX(std_X);
    this->setSTDY(std_Y);
    this->setSTDZ(std_Z);
    this->addObservationKpt(kpt1);
    this->updateMeanViewDir(T1);
}

MapPoint::MapPoint(int id, double X, double Y, double Z, double std_X, double std_Y, double std_Z, shared_ptr<KeyPoint2> kpt1, shared_ptr<KeyPoint2> kpt2, Mat T1, Mat T2)
{
    this->id = id;
    this->observation_cnt = 2;
    this->setCoordX(X);
    this->setCoordY(Y);
    this->setCoordZ(Z);
    this->setSTDX(std_X);
    this->setSTDY(std_Y);
    this->setSTDZ(std_Z);
    this->addObservationKpt(kpt1);
    this->addObservationKpt(kpt2);
    this->updateMeanViewDir(T1);
    this->updateMeanViewDir(T2);
}

MapPoint::~MapPoint()
{
    //TODO: Implement destructor
    //std::cout << "MapPoint Destroyed" << std::endl;
}

void MapPoint::iterateObservationCounter()
{
    std::unique_lock lock(this->mutex_observation_cnt);
    this->observation_cnt += 1;
}

void MapPoint::setCoordX(double X)
{
    std::unique_lock lock(this->mutex_coord);
    this->X = X;
}

void MapPoint::setCoordY(double Y)
{
    std::unique_lock lock(this->mutex_coord);
    this->Y = Y;
}

void MapPoint::setCoordZ(double Z)
{
    std::unique_lock lock(this->mutex_coord);
    this->Z = Z;
}

void MapPoint::setCoordXYZ(cv::Mat XYZ)
{
    std::unique_lock lock(this->mutex_coord);
    this->std_X = XYZ.at<double>(0,0);
    this->std_Y = XYZ.at<double>(1,0);
    this->std_Z = XYZ.at<double>(2,0);
}

void MapPoint::setSTDX(double std_X)
{
    std::unique_lock lock(this->mutex_coord_std);
    this->std_X = std_X;
}

void MapPoint::setSTDY(double std_Y)
{
    std::unique_lock lock(this->mutex_coord_std);
    this->std_Y = std_Y;
}

void MapPoint::setSTDZ(double std_Z)
{
    std::unique_lock lock(this->mutex_coord_std);
    this->std_Z = std_Z;
}

void MapPoint::update3DLocation( cv::Mat XYZ )
{
    /*
    Arguments:
        XYZ:    Dehomogenized 3D point in the world frame [shape 4 x n].
    */
    this->setCoordXYZ(XYZ);
}

void MapPoint::update3DUncertainty( cv::Mat XYZ_std )
{
    /*
    Arguments:
        XYZ_std:    Uncertainty of XYZ measurment. [shape 3 x n].
    TODO: Implement MapPoint::update3DUncertainty( cv::Mat XYZ_std ) function
    */
}

void MapPoint::updateMeanViewDir(cv::Mat T)
{
    /*
    Arguments:
        T:  Global extrinsic matrix of 1 and 2 [shape 4 x 4].
    */
    /* Iteratively updates the mean viewing direction based on all
       frames the point has been observed */
    // TODO: Implement MapPoint::updateMeanViewDir(KeyPoint2 new_kpt) function
}

void MapPoint::updateRepresentativeDescriptor(cv::Mat descriptor, std::string descr_type)
{
    /* Iteratively updates the representative descriptor of the map point*/
    // TODO: Implement MapPoint::updateRepresentativeDescriptor(std::string descr_type="orb") function
}

void MapPoint::addObservation(std::shared_ptr<KeyPoint2> kpt, cv::Mat XYZ, cv::Mat XYZ_std, cv::Mat T, std::string descr_type)
{
    /*
    Arguments:
        kpt:        Keypoint of new observation.
        XYZ:        Dehomogenized 3D point in the world frame [shape 4 x n].
        XYZ_std:    Uncertainty of XYZ measurment.
        T:          Global extrinsic matrix of 1 and 2 [shape 4 x 4].
        descr_type: Descriptor of interest of kpt. (This should maybe be copied from all entries of the descriptor list in kpt)
    */
    this->update3DLocation( XYZ );
    this->update3DUncertainty( XYZ_std );
    this->updateMeanViewDir( T );
    this->updateRepresentativeDescriptor( kpt->getDescriptor(descr_type), descr_type );
    this->addObservationKpt( kpt );
    this->iterateObservationCounter();
}

void MapPoint::addObservationKpt(shared_ptr<KeyPoint2> kpt)
{
    std::unique_lock lock(this->mutex_observation_keypoints);
    this->observation_keypoints[kpt->getObservationFrameNr()] = kpt;
}

// Read functions

int MapPoint::getId()
{
    std::shared_lock lock(this->mutex_id);
    return this->id;
}

int MapPoint::getObservationCounter()
{
    std::shared_lock lock(this->mutex_observation_cnt);
    return this->observation_cnt;
}

double MapPoint::getCoordX()
{
    std::shared_lock lock(this->mutex_coord);
    return this->X;
}

double MapPoint::getCoordY()
{
    std::shared_lock lock(this->mutex_coord);
    return this->Y;
}

double MapPoint::getCoordZ()
{
    std::shared_lock lock(this->mutex_coord);
    return this->Z;
}

double MapPoint::getSTDX()
{
    std::shared_lock lock(this->mutex_coord_std);
    return this->std_X;
}

double MapPoint::getSTDY()
{
    std::shared_lock lock(this->mutex_coord_std);
    return this->std_Y;
}

double MapPoint::getSTDZ()
{
    std::shared_lock lock(this->mutex_coord_std);
    return this->std_Z;
}

cv::Mat MapPoint::getCoordXYZ()
{
    cv::Mat XYZ1;
    XYZ1 = (cv::Mat_<double>(4,1)<<  this->getCoordX(),
                                    this->getCoordY(),
                                    this->getCoordZ(),
                                    1);
    return XYZ1;
}

Mat MapPoint::getMeanViewingDir()
{
    std::shared_lock lock(this->mutex_mean_view_dir);
    return this->mean_view_dir;
}

Mat MapPoint::getRepresentativeDescriptor(std::string descr_type)
{
    std::shared_lock lock(this->mutex_rep_descr);
    return this->rep_descr[descr_type];
}

weak_ptr<KeyPoint2> MapPoint::getObservationKpt(int kpt_frame_nr)
{
    std::shared_lock lock(this->mutex_observation_keypoints);
    return this->observation_keypoints[kpt_frame_nr];
}