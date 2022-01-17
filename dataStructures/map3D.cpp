

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
    std::unique_lock lock(this->mutex_map_points);
    this->map_points.push_back(map_point);
}

void Map3D::removeMapPoint(int idx)
{
    std::unique_lock lock(this->mutex_map_points);
    this->map_points.erase( this->map_points.begin() + idx );
}

void Map3D::batchUpdateMap(vector<shared_ptr<KeyPoint2>>& kpts1, vector<shared_ptr<KeyPoint2>>& kpts2, cv::Mat T1, cv::Mat T2, cv::Mat XYZ, cv::Mat XYZ_std)
{
    /*
    Arguments:
        kptsX:  List of matched keypoints,in order of matches [shape n]
        XYZ:    Dehomogenized 3D point in the world frame, ordered in same manner 
                as kptsX [shape 4 x n] 
    */
    // If existing track, register new kpt to track and update position
    // If new track, create new map point and register both kpts and give position

    int N = XYZ.cols;
    int frame1_nr = kpts1[0]->getObservationFrameNr();
    int frame2_nr = kpts2[0]->getObservationFrameNr();
    cv::Mat new_T;
    shared_ptr<MapPoint> temp_map_point;
    shared_ptr<KeyPoint2> temp_key_point1, temp_key_point2;
    vector<shared_ptr<KeyPoint2>> new_kpts, exist_kpts;


    // Identifying the new and existing keypoint lists
    //TODO: Find a more efficient way of doing this without copying arrays
    if ( frame1_nr > frame2_nr )
    {
        new_kpts = kpts1;
        exist_kpts = kpts2;
        new_T = T1;
    }
    else
    {
        new_kpts = kpts2;
        exist_kpts = kpts1;
        new_T = T2;
    }

    for ( int i = 0; i < N; i++ )
    {
        temp_key_point1 = exist_kpts[i];
        temp_key_point2 = new_kpts[i];
        if ( temp_key_point1->getMapPoint() == nullptr )
        {
            shared_ptr<MapPoint> map_point = std::make_shared<MapPoint>(XYZ.at<double>(0,i), 
                                                                        XYZ.at<double>(1,i), 
                                                                        XYZ.at<double>(2,i), 
                                                                        XYZ_std.at<double>(0,i), 
                                                                        XYZ_std.at<double>(1,i), 
                                                                        XYZ_std.at<double>(2,i), 
                                                                        kpts1[i], kpts2[i],
                                                                        T1, T2);
            this->addMapPoint(map_point);
            temp_key_point1->setMapPoint(map_point);
            //temp_key_point2->setMapPoint(map_point);
        }
        else
        {
            temp_map_point = temp_key_point1->getMapPoint();
            temp_map_point->addObservation( temp_key_point2, XYZ.col(i), XYZ_std.col(i), new_T );
            temp_key_point2->setMapPoint(temp_map_point);
        }
    }
}

void Map3D::resetMap()
{
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


// Static functions
void Map3D::calculateReprojectionError(cv::Mat uv1, cv::Mat uv2, cv::Mat K1, cv::Mat K2, cv::Mat T1, cv::Mat T2, cv::Mat reproj_error1, cv::Mat reproj_error2)
{
    /*
    Arguments: 
        uv:             Homogeneous pixel coordinates in image 1 and 2 [shape 3 x n].
        K:              Kalibration matrix for image 1 and 2 [shape 3 x 3].
        T:              Global extrinsic matrix of frame 1 and 2 [shape 4 x 4].
    Returns:
        reproj_error:   Reprojection error of uv1, uv2 [shape 2 x n]
    TODO:               Implement this function
    */
}

cv::Mat Map3D::calculate3DUncertainty(cv::Mat XYZ, cv::Mat uv1, cv::Mat uv2, cv::Mat K1, cv::Mat K2, cv::Mat T1, cv::Mat T2)
{
    /*
    Arguments: 
        XYZ:            Dehomogenized 3D point in the world frame [shape 4 x n].
        uv:             Homogeneous pixel coordinates in image 1 and 2 [shape 3 x n].
        K:              Kalibration matrix for image 1 and 2 [shape 3 x 3].
        T:              Extrinsic matrix between frame 1 and 2 [shape 4 x 4].
    Returns:
        XYZ_u:        Uncertainty of XYZ points in 3D [shape 3 x n]
    TODO:               Implement this function
    */

    return cv::Mat::zeros(3, XYZ.cols, CV_64F);
}