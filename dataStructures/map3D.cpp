

#include "map3D.hpp"

using std::vector;
using std::shared_ptr;

Map3D::Map3D()
{
    //TODO: Implement constructor
}

Map3D::~Map3D()
{
    //TODO: Implement destructor
}


// Write functions
void Map3D::addMapPoint(std::shared_ptr<MapPoint> map_point)
{
    /* Add <map_point> to <Map3D> */
    
    std::unique_lock lock(this->mutex_map_points);
    this->map_points.push_back(map_point);
}

void Map3D::createMapPoint(cv::Mat XYZ, cv::Mat XYZ_std, 
                            shared_ptr<KeyPoint2> kpt1, cv::Mat T1)
{
    /*
    Creates a map point based on 3D information.

    Arguments:
        XYZ:        Dehomogenized 3D point in the world frame, ordered in same 
                    manner as kptsX [shape 4 x 1] 
        XYZ_std:    Standard deviation of 3D point in the world frame, same 
                    structure as XYZ [shape 3 x 1] 
        kptX:       Keypoint associated with the new mapPoint
        TX:         Global transformation matrices of frameX
    Overview:
        Generates a map point based on only one point (with 3D data)
    */
    int map_point_id = this->getNumMapPoints();
    shared_ptr<MapPoint> map_point = std::make_shared<MapPoint>(map_point_id,
                                                                XYZ.at<double>(0,0), 
                                                                XYZ.at<double>(1,0), 
                                                                XYZ.at<double>(2,0), 
                                                                XYZ_std.at<double>(0,0), 
                                                                XYZ_std.at<double>(1,0), 
                                                                XYZ_std.at<double>(2,0), 
                                                                kpt1, T1);
    this->addMapPoint(map_point);
    kpt1->setMapPoint(map_point);
}

void Map3D::removeMapPoint(int idx)
{
    std::unique_lock lock(this->mutex_map_points);
    this->map_points.erase( this->map_points.begin() + idx );
}

void Map3D::updateMap(shared_ptr<KeyPoint2> kpt1, shared_ptr<KeyPoint2> kpt2, 
                        cv::Mat T1, cv::Mat T2, cv::Mat XYZ1, cv::Mat XYZ_std)
{
    /*
    Creates new <MapPoint> if a match belong to new track and updates 
        <MapPoint> if match belongs to existing track.

    Arguments:
        kptX:       Matched keypoints, kpt1 is the new keypoint, kpt2 is the 
                    old keypoint.
        TX:         Global transformation matrices of frameX.
        XYZ1:       Dehomogenized 3D point in the world frame, ordered in same 
                    manner as kptsX [shape 4 x 1].
        XYZ_std:    Standard deviation of 3D point in the world frame, same 
                    structure as XYZ [shape 3 x 1].
    */

    shared_ptr<MapPoint> temp_map_point;
    int map_point_id = this->getNumMapPoints();

    if ( kpt2->getMapPoint() == nullptr )
    {
        shared_ptr<MapPoint> map_point = std::make_shared<MapPoint>(map_point_id,
                                                                    XYZ1.at<double>(0,0), 
                                                                    XYZ1.at<double>(1,0), 
                                                                    XYZ1.at<double>(2,0), 
                                                                    XYZ_std.at<double>(0,0), 
                                                                    XYZ_std.at<double>(1,0), 
                                                                    XYZ_std.at<double>(2,0), 
                                                                    kpt1, kpt2,
                                                                    T1, T2);
        this->addMapPoint(map_point);
        kpt1->setMapPoint(map_point);
        kpt2->setMapPoint(map_point);
    }
    else
    {
        temp_map_point = kpt2->getMapPoint();
        temp_map_point->addObservation( kpt1, XYZ1, XYZ_std, T1 );
        kpt1->setMapPoint(temp_map_point);
    }
}

void Map3D::batchUpdateMap( vector<shared_ptr<KeyPoint2>>& kpts1, 
                            vector<shared_ptr<KeyPoint2>>& kpts2, 
                            cv::Mat T1, cv::Mat T2, 
                            cv::Mat XYZ1, cv::Mat XYZ_std)
{
    /*
    Creates new <MapPoint> if a match belong to new track and updates 
        <MapPoint> if match belongs to existing track.

    Arguments:
        kptsX:      List of matched keypoints,in order of matches [shape n]
        TX:         Global transformation matrices of frameX
        XYZ:        Dehomogenized 3D point in the world frame, ordered in same 
                    manner as kptsX [shape 4 x n] 
        XYZ_std:    Standard deviation of 3D point in the world frame, same 
                    structure as XYZ [shape 3 x n] 
    */

    int N = XYZ1.cols;

    for ( int i = 0; i < N; ++i )
    {
        this->updateMap(kpts1[i], kpts2[i], T1, T2, 
                        XYZ1.col(i), XYZ_std.col(i));
    }
}

void Map3D::resetMap()
{
    /* Removes all <MapPoints> in <Map3D> */
    std::unique_lock lock(this->mutex_map_points);
    this->map_points.clear();
}



// Read functions

int Map3D::getNumMapPoints()
{
    std::shared_lock lock(this->mutex_map_points);
    return this->map_points.size();
}

std::shared_ptr<MapPoint> Map3D::getMapPoint(int idx)
{
    std::shared_lock lock(this->mutex_map_points);
    return this->map_points[idx];
}

vector<std::shared_ptr<MapPoint>> Map3D::getAllMapPoints()
{
    std::shared_lock lock(this->mutex_map_points);
    return this->map_points;
}

cv::Mat Map3D::compileMapPointLocs()
{
    std::shared_lock(this->mutex_map_points);
    int N = this->getNumMapPoints();
    cv::Mat map_points_loc = cv::Mat::zeros(4, N, CV_64F);

    for ( int i = 0; i < N; ++i )
    {
        this->getMapPoint(i)->getCoordXYZ1().copyTo(map_points_loc.col(i));
    }
    return map_points_loc;
}

vector<cv::Point3f> Map3D::compileCVPoints3f()
{
    std::shared_lock lock(this->mutex_map_points);
    int N = this->getNumMapPoints();
    std::vector<cv::Point3f> map_points3f( N );

    #pragma omp parallel for
    for ( int i = 0; i < N; ++i )
    {
        map_points3f[i] = this->getMapPoint(i)->compileCVPoint3f();
    }
    return map_points3f;
}

vector<cv::Point3d> Map3D::compileCVPoints3d()
{
    std::shared_lock lock(this->mutex_map_points);
    int N = this->getNumMapPoints();
    std::vector<cv::Point3d> map_points3d( N );

    #pragma omp parallel for
    for ( int i = 0; i < N; ++i )
    {
        map_points3d[i] = this->getMapPoint(i)->compileCVPoint3d();
    }
    return map_points3d;
}


// Static functions
void Map3D::calculateReprojectionError( cv::Mat uv1, cv::Mat uv2, cv::Mat K1, 
                                        cv::Mat K2, cv::Mat T1, cv::Mat T2, 
                                        cv::Mat reproj_error1, 
                                        cv::Mat reproj_error2)
{
    /*
    Arguments: 
        uv:             Homogeneous pixel coordinates in image 1 and 2 
                        [shape 3 x n].
        K:              Kalibration matrix for image 1 and 2 [shape 3 x 3].
        T:              Global extrinsic matrix of frame 1 and 2 [shape 4 x 4].
    Returns:
        reproj_error:   Reprojection error of uv1, uv2 [shape 2 x n]
    TODO:               Implement this function
    */
}

cv::Mat Map3D::calculate3DUncertainty(  cv::Mat XYZ, 
                                        cv::Mat uv1, cv::Mat uv2, 
                                        cv::Mat K1, cv::Mat K2, 
                                        cv::Mat T1, cv::Mat T2)
{
    /*
    Arguments: 
        XYZ:            Dehomogenized 3D point in the world frame [shape 4 x n].
        uv:             Homogeneous pixel coordinates in image 1 and 2 
                        [shape 3 x n].
        K:              Kalibration matrix for image 1 and 2 [shape 3 x 3].
        T:              Extrinsic matrix between frame 1 and 2 [shape 4 x 4].
    Returns:
        XYZ_u:        Uncertainty of XYZ points in 3D [shape 3 x n]
    TODO:               Implement this function
    */

    return cv::Mat::zeros(3, XYZ.cols, CV_64F);
}