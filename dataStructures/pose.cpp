#include <map>
#include <vector>
#include <memory>
#include <fstream>
#include <iostream>
#include <sys/stat.h>
#include <opencv2/opencv.hpp>

using std::shared_ptr;
using std::weak_ptr;
using std::vector;
using std::map;
using cv::Mat;

#include "parametrization.hpp"
#include "pose.hpp"

Pose::Pose(Mat E_matrix, shared_ptr<FrameData> frame1, shared_ptr<FrameData> frame2, ParamID parametrization_id, int pose_nr)
{
    this->pose_nr = pose_nr;
    this->frame1 = frame1;
    this->frame2 = frame2;

    this->setPose( E_matrix, parametrization_id );
}

Pose::~Pose()
{
    //TODO: Implement destructor
    //std::cout << "POSE DESTROYED: " << this->pose_nr << std::endl;
}


// Write functions
void Pose::setTMatrix(Mat T_matrix)
{
    /* Needs to update all other values that are dependent on this one,
       or set a flag
       Warning: Not protected  */
    this->T_matrix = T_matrix;
}

void Pose::setRMatrix(Mat R_matrix)
{
    /* Needs to update all other values that are dependent on this one,
       or set a flag
       Warning: Not protected  */
    this->R_matrix = R_matrix;
}

void Pose::settvector(Mat t_vector)
{
    /* Needs to update all other values that are dependent on this one,
       or set a flag
       Warning: Not protected  */
    this->t_vector = t_vector;
}

void Pose::setEmatrix(Mat E_matrix)
{
    /* Needs to update all other values that are dependent on this one,
       or set a flag
       Warning: Not protected  */
    this->E_matrix = E_matrix;
}


void Pose::setPose( Mat E_matrix, ParamID parametrization_id )
{
    /* Updating pose variables on the basis of <E_matrix> 
       NOTE: It is slow because it projects the points in 3d space
              but throws these away without using them 
       TODO: Do the cheriality tests manually such that it can be 
              based off of stored trianglated points*/
    Mat R, t;
    this->decomposeEMatrixSlow(E_matrix, R, t);
    this->updatePoseVariables(E_matrix, R, t);
}

void Pose::setPose( vector<double> params, ParamID parametrization_id )
{
    /* Updating pose variables on the basis of <params> 
       Function is inefficient as it triangulates points and checks
       cheriality test without returning these*/

    this->setParametrization( params, parametrization_id );
    this->updatePoseVariables(parametrization_id);
}

void Pose::updateParametrization(ParamID parametrization_id)
{
    /* Sets the correct parametrization variables based on pose object internal parameters
        NOTE: Not neccessary in most cases, just for easier human comsumption of parameters */
    std::unique_lock lock(this->mutex_params_map);
    std::shared_ptr<Parametrization> temp_parametrization;

    if ( this->params.find(parametrization_id) == this->params.end())
    {
        this->createParametrization( this->getRMatrix(), this->gettvector(), parametrization_id );
    }
    else
    {
        temp_parametrization = this->getParametrization( parametrization_id );
        temp_parametrization->setParams( this->getRMatrix(), this->gettvector() );
    }
}

void Pose::setParametrization( vector<double> params, ParamID parametrization_id )
{
    /* Sets the correct parametrization variables based on <params> and <parametrization_id.
       Also creates a new parametrization if type defined by <parametrization_id> does not
       exist
       WARNING: This does not make sure that the pose object is also updated, USE WITH CARE */

    std::unique_lock lock(this->mutex_params_map);
    std::shared_ptr<Parametrization> temp_parametrization;

    if ( this->params.find(parametrization_id) == this->params.end())
    {
        this->createParametrization( params, parametrization_id );
    }
    else
    {
        temp_parametrization = this->getParametrization( parametrization_id );
        this->getParametrization( parametrization_id )->setParams( params );
    }
}

void Pose::createParametrization( vector<double> params, ParamID parametrization_id )
{
    // Warning: Not directly protected against race conditions, use <setParametrization>

    if ( parametrization_id == ParamID::STDPARAM)
    {
        this->params[parametrization_id] = shared_ptr<Parametrization>(new StdParam(params));
    }
    else
    {
        std::cout << "Error: Parametrization initializer for <parametrization_id> not specified" << std::endl;
    }
}

void Pose::createParametrization( Mat R, Mat t, ParamID parametrization_id )
{
    // Warning: Not directly protected against race conditions, use <setParametrization>

    if ( parametrization_id == ParamID::STDPARAM)
    {
        this->params[parametrization_id] = shared_ptr<Parametrization>(new StdParam(R, t));
    }
    else
    {
        std::cout << "Error: Parametrization initializer for <parametrization_id> not specified" << std::endl;
    }
}

void Pose::updatePoseVariables(Mat R_matrix, Mat t_vector)
{
    /* Updates pose based on R and t */
    std::unique_lock lock(mutex_pose);
    this->setRMatrix( R_matrix );
    this->settvector( t_vector );
    this->setTMatrix( this->composeTMatrix( R_matrix, t_vector ) );
    this->setEmatrix( this->composeEMatrix(R_matrix, t_vector) );
    this->riseParametrizationInvalidFlags();
}

void Pose::updatePoseVariables(Mat E_matrix, Mat R_matrix, Mat t_vector)
{
    /* Updates pose based on E, R and t */
    std::unique_lock lock(mutex_pose);
    this->setRMatrix( R_matrix );
    this->settvector( t_vector );
    this->setTMatrix( this->composeTMatrix( R_matrix, t_vector ) );
    this->setEmatrix( E_matrix );
    this->riseParametrizationInvalidFlags();
}

void Pose::updatePoseVariables(ParamID parametrization_id)
{
    /* Updates pose based on values in <Parametrization> */
    std::unique_lock lock(this->mutex_pose);
    std::shared_ptr<Parametrization> temp_parametrization;
    temp_parametrization = this->getParametrization( parametrization_id );

    this->setRMatrix( temp_parametrization->composeRMatrix() );
    this->settvector( temp_parametrization->composeTransVec() );
    this->setTMatrix( this->composeTMatrix( this->getRMatrix(), this->gettvector() ) );
    this->setEmatrix( this->composeEMatrix( this->getRMatrix(), this->gettvector() ) );
}

void Pose::riseParametrizationInvalidFlags(ParamID exeption)
{
    std::shared_lock lock(this->mutex_params_map);
    std::shared_ptr<Parametrization> temp_parametrization;
    map<ParamID, shared_ptr<Parametrization>>::iterator it;

    for (it = this->params.begin(); it != this->params.end(); it++)
    {
        if (it->first != exeption)
        {
            temp_parametrization = it->second;
            temp_parametrization->setValidFlag(false);
        }
    }
}


// Read functions
int Pose::getPoseNr()
{
    return this->pose_nr;
}

shared_ptr<FrameData> Pose::getFrame1()
{
    return this->frame1.lock();
}

shared_ptr<FrameData> Pose::getFrame2()
{
    return this->frame2.lock();
}

Mat Pose::getTMatrix()
{
    std::shared_lock lock(this->mutex_pose);
    return this->T_matrix;
}

Mat Pose::getRMatrix()
{
    std::shared_lock lock(this->mutex_pose);
    return this->R_matrix;
}

Mat Pose::gettvector()
{
    std::shared_lock lock(this->mutex_pose);
    return this->t_vector;
}

Mat Pose::getEMatrix()
{
    std::shared_lock lock(this->mutex_pose);
    return this->E_matrix;
}

shared_ptr<FrameData> Pose::getConnectingFrame(int connecting_frame_nr)
{
    std::shared_ptr<FrameData> temp_frame1, temp_frame2;
    temp_frame1 = this->getFrame1();
    temp_frame2 = this->getFrame2();

    if ( temp_frame1->getFrameNr() == connecting_frame_nr )
    {
        return temp_frame1;
    }
    else if ( temp_frame2->getFrameNr() == connecting_frame_nr )
    {
        return temp_frame2;
    }
    else
    {
        std::cout << "ERROR: <connecting_frame_nr> does not hold a frame connected to this pose" << std::endl;
        return nullptr;
    }
}

shared_ptr<Parametrization> Pose::getParametrization( ParamID parametrization_id )
{
    std::shared_lock lock(this->mutex_params_map);
    return this->params[parametrization_id];
}

void Pose::decomposeEMatrixSlow(Mat &E_matrix, Mat &R, Mat &t)
{
    /* Function is inefficient as it traingulates points and checks 
       cheriality test without returning these
       NOTE: Assumes both frames uses the same camera matrix */

    int num_points, num_points1, num_points2;
    std::shared_ptr<FrameData> temp_frame1, temp_frame2;
    temp_frame1 = this->getFrame1();
    temp_frame2 = this->getFrame2();

    vector<cv::Point> pts1, pts2;
    compileMatchedCVPoints(temp_frame1, temp_frame2, pts1, pts2);
    num_points = cv::recoverPose( E_matrix, pts1, pts2, temp_frame1->getKMatrix(), R, t );
}

void Pose::write2File( std::string file_path, ParamID parametrization_id )
{
    /* Writing relative pose to "output/poses.txt" file
       TODO: Check if file_path folder exists, if it does not, create it,
       Maybe create a dataStructures utils file for this.*/
    
    std::ofstream pose_file;
    pose_file.open(file_path, std::ios_base::app);
    if (pose_file.is_open())
    {
        pose_file << *getParametrization( parametrization_id );
        pose_file << "\n";
        pose_file.close();
    }
    else
    {
        std::cout << "Unable to open file: " << file_path << std::endl;
    }
}


// Static functions
Mat Pose::composeTMatrix(Mat R, Mat t)
{
    Mat T =    (cv::Mat_<double>(4,4) <<
                    R.at<double>(0,0),  R.at<double>(0,1),  R.at<double>(0,2),      t.at<double>(0,0),
                    R.at<double>(1,0),  R.at<double>(1,1),  R.at<double>(1,2),      t.at<double>(1,0),
                    R.at<double>(2,0),  R.at<double>(2,1),  R.at<double>(2,2),      t.at<double>(2,0),
                    0,                  0,                  0,                      1                   );
    return T;
}

Mat Pose::composeEMatrix(Mat R, Mat t)
{
    /* Composes the E matrix from the R and t matrixes */
    Mat t_skew =    (cv::Mat_<double>(3,3) <<
                    0,                  -t.at<double>(2,0),  t.at<double>(1,0),
                    t.at<double>(2,0),   0,                  -t.at<double>(0,0),
                    -t.at<double>(1,0),  t.at<double>(0,0),   0
                    );
    // Matrix multipliation
    Mat E = t_skew * R;
    return E;
}


// Operator overload
std::ostream& operator<<(std::ostream& os, const Pose& obj)
{
    os << "Printing POSE";
    return os;
}