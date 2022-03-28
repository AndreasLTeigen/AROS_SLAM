#include <math.h>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>


#include "parametrization.hpp"

using std::vector;
using cv::Mat;


//class Parametrization

Parametrization::Parametrization(ParamID parametrization_id, bool valid)
{
    this->parametrization_id = parametrization_id;
    this->valid = valid;
}

Parametrization::~Parametrization()
{
    //TODO: Implement destructor
}


// Write functions
void Parametrization::setValidFlag(bool value)
{
    std::unique_lock lock(this->mutex_valid_flag);
    this->valid = value;
}


// Read functions
bool Parametrization::isValid()
{
    std::shared_lock lock(this->mutex_valid_flag);
    return this->valid;
}


// Static functions
bool Parametrization::isRotationMatrix(cv::Mat &R)
{
    //Checks if <R> is a valid rotation matrix
    cv::Mat Rt;
    transpose(R, Rt);
    cv::Mat shouldBeIdentity = Rt * R;
    cv::Mat I = cv::Mat::eye(3,3, shouldBeIdentity.type());
    return  cv::norm(I, shouldBeIdentity) < 1e-6;
}


// Operator overload
std::ostream& operator<<(std::ostream& out, Parametrization& obj)
{
    return obj.print(out);
}



//--------------------------------STANDARD PARAMETRIZATION-----------------------------------

StdParam::StdParam()
    :Parametrization(ParamID::STDPARAM, false)
{
    this->rx = -1;
    this->ry = -1;
    this->rz = -1;
    this->tx = -1;
    this->ty = -1;
    this->tz = -1;
}

StdParam::StdParam(vector<double> params, bool valid)
    :Parametrization(ParamID::STDPARAM, valid)
{
    if (params.size() != 6)
    {
        std::cout << "Warning: Parameter vector not correct length" << std::endl;
    }
    this->rx = params[0];
    this->ry = params[1];
    this->rz = params[2];
    this->tx = params[3];
    this->ty = params[4];
    this->tz = params[5];
}

StdParam::StdParam(double rx, double ry, double rz, double tx, double ty, double tz, bool valid)
    :Parametrization(ParamID::STDPARAM, valid)
{
    this->rx = rx;
    this->ry = ry;
    this->rz = rz;
    this->tx = tx;
    this->ty = ty;
    this->tz = tz;
}

StdParam::StdParam(Mat R, Mat t, bool valid)
    :Parametrization(ParamID::STDPARAM, valid)
{
    this->decomposeRMatrix(R);
    this->decomposeTVector(t);
}

StdParam::~StdParam()
{
    //TODO:Implement destructor
}


// Write functions
void StdParam::setParams(vector<double> params)
{
    std::unique_lock lock(this->mutex_parametrization);
    if (params.size() != 6)
    {
        std::cout << "Warning: Parameter vector not correct length" << std::endl;
    }
    this->rx = params[0];
    this->ry = params[1];
    this->rz = params[2];
    this->tx = params[3];
    this->ty = params[4];
    this->tz = params[5];
    this->setValidFlag(true);
}

void StdParam::setParams( Mat R, Mat t )
{
    this->decomposeRMatrix(R);
    this->decomposeTVector(t);
    this->setValidFlag(true);
}

void StdParam::decomposeRMatrix( Mat &R )
{
    std::unique_lock lock(this->mutex_parametrization);
    assert(isRotationMatrix(R));

    double sy = sqrt(R.at<double>(0,0) * R.at<double>(0,0) +  R.at<double>(1,0) * R.at<double>(1,0) );

    bool singular = sy < 1e-6; // If

    double x, y, z;
    if (!singular)
    {
        this->rx = atan2(R.at<double>(2,1) , R.at<double>(2,2));
        this->ry = atan2(-R.at<double>(2,0), sy);
        this->rz = atan2(R.at<double>(1,0), R.at<double>(0,0));
    }
    else
    {
        this->rx = atan2(-R.at<double>(1,2), R.at<double>(1,1));
        this->ry = atan2(-R.at<double>(2,0), sy);
        this->rz = 0;
    }
}

void StdParam::decomposeTVector( Mat t )
{
    std::unique_lock lock(this->mutex_parametrization);
    this->tx = t.at<double>(0,0);
    this->ty = t.at<double>(1,0);
    this->tz = t.at<double>(2,0);
}


// Read functions
vector<double> StdParam::getRotParams()
{
    /* Returns rotational parameters in a single vector */
    std::shared_lock lock(this->mutex_parametrization);
    vector<double> ret_vec{ this->rx, this->ry, this->rz };
    return ret_vec;
}

vector<double> StdParam::getTransParams()
{
    /* Returns translational parameters in a single vector */
    std::shared_lock lock(this->mutex_parametrization);
    vector<double> ret_vec{ this->tx, this->ty, this->tz };
    return ret_vec;
}

vector<double> StdParam::getParamVector()
{
    /* Returns all parameters in a single vector */
    std::shared_lock lock(this->mutex_parametrization);
    vector<double> ret_vec{ this->rx, this->ry, this->rz,
                            this->tx, this->ty, this->tz };
    return ret_vec;
}

Mat StdParam::composeRMatrix()
{
    vector<double> r_vec = this->getRotParams();

    // Calculate rotation about x axis
    Mat R_x = (cv::Mat_<double>(3,3) <<
               1,               0,              0,
               0,               cos(r_vec[0]),  -sin(r_vec[0]),
               0,               sin(r_vec[0]),  cos(r_vec[0])
               );
    
    // Calculate rotation about y axis
    Mat R_y = (cv::Mat_<double>(3,3) <<
               cos(r_vec[1]),   0,              sin(r_vec[1]),
               0,               1,              0,
               -sin(r_vec[1]),  0,              cos(r_vec[1])
               );
    
    // Calculate rotation about z axis
    Mat R_z = (cv::Mat_<double>(3,3) <<
               cos(r_vec[2]),   -sin(r_vec[2]), 0,
               sin(r_vec[2]),   cos(r_vec[2]),  0,
               0,               0,              1
               );
    
    // Combined rotation matrix
    Mat R = R_z * R_y * R_x;

    return R;
}

Mat StdParam::composeTransVec()
{
    vector<double> t_vec = this->getTransParams();

    Mat t = (cv::Mat_<double>(3,1) <<
            t_vec[0],
            t_vec[1],
            t_vec[2]
            );
    return t;
}

std::ostream& StdParam::print(std::ostream& out)
{
    /* Returns pose: rx ry rz tx ty tz*/
    vector<double> paramVector = getParamVector();
    return out << paramVector[0] << " " << paramVector[1] << " " 
                << paramVector[2] << " " << paramVector[3] << " " 
                << paramVector[4] << " " << paramVector[5];
}



//Static functions
void StdParam::composeRMatrixAndTParam(std::vector<double>& param, cv::Mat& R, cv::Mat& t)
{
    /*
    Arguments:
        param:  rx, ry, rz, tx, ty, tz [1 x 6].
    Returns:
        R:      Rotation matrix [3 x 3].
        t:      Translation vector [3 x 1].
    */

    // Calculate rotation about x axis
    Mat R_x = (cv::Mat_<double>(3,3) <<
                1,               0,              0,
                0,               cos(param[0]),  -sin(param[0]),
                0,               sin(param[0]),  cos(param[0])
                );

    // Calculate rotation about y axis
    Mat R_y = (cv::Mat_<double>(3,3) <<
                cos(param[1]),   0,              sin(param[1]),
                0,               1,              0,
                -sin(param[1]),  0,              cos(param[1])
                );

    // Calculate rotation about z axis
    Mat R_z = (cv::Mat_<double>(3,3) <<
                cos(param[2]),   -sin(param[2]), 0,
                sin(param[2]),   cos(param[2]),  0,
                0,               0,              1
                );

    // Combined rotation matrix
    R = R_z * R_y * R_x;


    t = (cv::Mat_<double>(3,1) <<
            param[3],
            param[4],
            param[5]
            );
}