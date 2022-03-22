#include <fstream>
#include <sstream>
#include <iostream>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "util.hpp"
#include "yaml-cpp/yaml.h"

using std::string;


std::string zeroPad(int num, int pad_n)
{
    /*
    Arguments:
        num:        Integer one wants to add padding to.
        pad_n:      Total length after padding.
    Returns:
        out:        String of integer <num> padded with 0s to be a total of <pad_n> long.
    */
    std::ostringstream out;
    out << std::setfill('0') << std::setw(pad_n) << num;
    return out.str();
}

void reduceImgContrast(cv::Mat img, int lower_level, int upper_level)
{
    /* Function for moving the contrast of the image to the newly defined
       range <upper_bound - lower_bound> 
       Note: This function results in throwing the [WARN:0] at runtime */
    img = ( img * lower_level / upper_level ) + lower_level;
}

void drawCircle(cv::Mat &img, cv::Point point, int radius)
{
    cv::Scalar color_red = cv::Scalar( 0, 0, 255, 128 );
    std::cout << "Drawing circle at: " << point << std::endl;
    cv::circle(img, point, radius, color_red, 2);
}

void drawCircle(cv::Mat &img, cv::Mat point_mat, int radius)
{
    cv::Point point = cv::Point(point_mat.at<double>(0), point_mat.at<double>(1));
    drawCircle(img, point, radius);
}

cv::Mat fundamentalFromEssential(cv::Mat E_matrix, cv::Mat K_matrix)
{
    cv::Mat K_inv = K_matrix.inv();
    cv::Mat F = K_inv.t()*E_matrix*K_inv;
    return F;
}

cv::Mat calculateEpipole(cv::Mat E_matrix)
{
    /* Caclulates the epipole of in frame 2 the essential matrix of the
       transformation between frame 1 and frame 2, F_matrix can also
       be used as input */

    cv::Mat epipole;
    Eigen::Matrix<double, 3, 3> E_matrix_eigen;
    cv::cv2eigen(E_matrix.t(), E_matrix_eigen);
    Eigen::FullPivLU<Eigen::Matrix<double, 3, 3>> lu(E_matrix_eigen);
    Eigen::Matrix<double, 3, 1> epipole_eigen = lu.kernel();
    cv::eigen2cv(epipole_eigen, epipole);
    epipole = epipole/epipole.at<double>(2);
    return epipole;
}

void drawEpipolarLines(cv::Mat F, cv::Mat &img_disp2,
                        std::vector<cv::Point> points1,
                        std::vector<cv::Point> points2)
{
    std::vector<cv::Vec<double,3>> epilines1;
    cv::computeCorrespondEpilines(points1, 1, F, epilines1); //Index starts with 1

    cv::RNG rng(0);
    for(size_t i=0; i<points1.size(); i++)
    {
        /*
        * Epipolar lines of the 1st point set are drawn in the 2nd image and vice-versa
        */
        cv::Scalar color(rng.uniform(0,256),rng.uniform(0,256),rng.uniform(0,256));
        cv::line(img_disp2,
            cv::Point(0,-epilines1[i][2]/epilines1[i][1]),
            cv::Point(img_disp2.cols,-(epilines1[i][2]+epilines1[i][0]*img_disp2.cols)/epilines1[i][1]),
            color);
    }
}

cv::Mat compileKMatrix( double fx, double fy, double cx, double cy )
{
    // Compiles the rotation matrix from the parameters <fx, fy, cx, cy>
    cv::Mat K;
    K = (cv::Mat_<double>(3,3)<<fx,  0,  cx,
                                0,  fy,  cy,
                                0,  0,    1);
    return K;
}

bool isRotationMatrix(cv::Mat &R)
{
    //Checks if <R> is a valid rotation matrix
    cv::Mat Rt;
    transpose(R, Rt);
    cv::Mat shouldBeIdentity = Rt * R;
    cv::Mat I = cv::Mat::eye(3,3, shouldBeIdentity.type());
    return  cv::norm(I, shouldBeIdentity) < 1e-6;
}

std::vector<double> rotationMatrixToEulerAngles(cv::Mat &R)
{
    assert(isRotationMatrix(R));

    double sy = sqrt(R.at<double>(0,0) * R.at<double>(0,0) +  R.at<double>(1,0) * R.at<double>(1,0) );

    bool singular = sy < 1e-6; // If

    double rx, ry, rz;
    if (!singular)
    {
        rx = atan2(R.at<double>(2,1) , R.at<double>(2,2));
        ry = atan2(-R.at<double>(2,0), sy);
        rz = atan2(R.at<double>(1,0), R.at<double>(0,0));
    }
    else
    {
        rx = atan2(-R.at<double>(1,2), R.at<double>(1,1));
        ry = atan2(-R.at<double>(2,0), sy);
        rz = 0;
    }
    return std::vector<double> {rx, ry, rz};
}

cv::Mat T2Rot(cv::Mat &T)
{
    // Check if getting rows and cols is quicker than this process
    cv::Mat rot = cv::Mat::zeros(3,3,CV_64F);
    rot = (cv::Mat_<double>(3,3)<<T.at<double>(0,0), T.at<double>(0,1), T.at<double>(0,2),
                                T.at<double>(1,0), T.at<double>(1,1), T.at<double>(1,2),
                                T.at<double>(2,0), T.at<double>(2,1), T.at<double>(2,2));
    return rot;
}

cv::Mat T2Trans(cv::Mat &T)
{
    // Chekc if getting rows and cols is quicker than this process
    cv::Mat trans = cv::Mat::zeros(3,1,CV_64F);
    trans = (cv::Mat_<double>(3,1)<<T.at<double>(0,3),
                                T.at<double>(1,3),
                                T.at<double>(2,3));
    return trans;
}

void T2RotAndTrans(cv::Mat &T, cv::Mat &R, cv::Mat &t)
{
    /*@TODO: Add an assertion to the dimensions of R and t*/

    R = T2Rot(T);

    t = T2Trans(T);
}

cv::Mat inverseTMatrix(cv::Mat T)
{
    cv::Mat T_inv = cv::Mat::eye(4,4,CV_64F);
    cv::Mat R(3,3,CV_64F);
    cv::Mat t(3,1,CV_64F);
    T2RotAndTrans(T, R, t);
    T_inv.rowRange(0,3).colRange(0,3) = R.t();
    T_inv.rowRange(0,3).col(3) = -R.t() * t;
    return T_inv;
}

cv::Mat compileTMatrix(std::vector<double> pose)
{
    /*
    Arguments:
        pose:       Vector containing all values of T [shape 1 x 12]
    Returns:
        T:          Transformation matrix [shape 4 x 4]
    */

    cv::Mat T = cv::Mat::eye(4,4,CV_64F);
    int h = 3;
    int w = 4;
    for ( int i = 0; i < h; i++ )
    {
        for ( int j = 0; j < w; j++ )
        {
            T.at<double>(i,j) = pose[i*w + j];
        }
    }
    return T;
}

cv::Mat xy2Mat(double x, double y)
{
    double data [2] = {x, y};
    cv::Mat xy = cv::Mat(1,2, CV_64F, data);
    return xy.t(); 
}

cv::Mat xyToxy1(double x, double y)
{
    cv::Mat xy = xy2Mat(x, y);
    homogenizeArray(xy);
    return xy;
}

void homogenizeArray(cv::Mat& arr)
{
    cv::Mat row = cv::Mat::ones(1, arr.cols, CV_64F);
    arr.push_back(row);
}

void dehomogenizeMatrix(cv::Mat& X)
{
    int num_rows = X.rows;

    #pragma omp parallel for
    for ( int i = 0; i < X.cols; i++ )
    {
        X.col(i) = X.col(i) / X.at<double>(num_rows-1,i);
    }
}

cv::Mat fitQuadraticForm(cv::Mat& x, cv::Mat& y, cv::Mat& z)
{

    /*
    Arguments:
        x:      List of x coordinates,  [N x 1].
        y:      List of y coordinates,  [N x 1].
        z:      List of z coordinates,  [N x 1].
    Returns:
        A:      Quadratic form fitted to (x, y, 1).T * A * (x, y, 1) = z, [3 x 3].
                   [[a11,   a12/2,  a1/2],
                    [a12/2, a22     a2/2],
                    [a1/2,  a2/2,   a0  ]]
    */

    // Reconstructing the problem into a linear least squares problem: z = D * a
    // D_i = [x^2, y^2, x*y, x, y, 1] and a = [a11, a22, a12, a1, a2, a0].T

    int N = z.rows;
    double x_i, y_i;
    cv::Mat D = cv::Mat::ones(N,6,CV_64F);

    for ( int i = 0; i < N; ++i )
    {
        x_i = x.at<double>(i,0);
        y_i = y.at<double>(i,0);
        D.at<double>(i, 0) = x_i*x_i;
        D.at<double>(i, 1) = y_i*y_i;
        D.at<double>(i, 2) = x_i*y_i;
        D.at<double>(i, 3) = x_i;
        D.at<double>(i, 4) = y_i;
    }

    cv::Mat a;
    cv::solve(D, z, a, cv::DECOMP_SVD);
    //std::cout << "Quadratic form fitted: " << a << std::endl;
    cv::Mat A = (cv::Mat_<double>(3,3)<<a.at<double>(0,0),      a.at<double>(0,2)/2,    a.at<double>(0,3)/2,
                                    a.at<double>(0,2)/2,    a.at<double>(0,1),      a.at<double>(0,4)/2,
                                    a.at<double>(0,3)/2,    a.at<double>(0,4)/2,    a.at<double>(0,5));

    return A;
}

cv::Mat dilateKptWDepth(cv::Mat xy1, double Z, cv::Mat T_wc, cv::Mat K)
{
    /*
    Arguments:
        xy1:    Homogeneous pixel coordinates [shape 3 x 1].
        Z:      Depth in meters from camera.
        T_wc:   Camera global transformation matrix [shape 4 x 4].
        K:      Camera intrinsic parameters [shape 3 x 3].
    Returns:
        XYZ1:    Position of map point given in global, homogeneous coordinates.
    Overview:
        XYZ = Z * (K^-1) * xy1 = Z * xy1_z
    */
    cv::Mat xy1_z; // TODO: K inverse should be computed on forhand.
    cv::solve(K, xy1, xy1_z);
    cv::Mat XYZ_c = Z * xy1_z;
    homogenizeArray(XYZ_c);
    cv::Mat XYZ1 = T_wc*XYZ_c;
    //std::cout << "T_wc:\n" << T_wc << std::endl;            //TODO: Remove 
    //std::cout << "XYZ1:\n" << XYZ1 << std::endl;            //TODO: Remove
    return XYZ1;
    //return cv::Mat::zeros(3, 1, CV_64F);
}

cv::Mat projectKpt(cv::Mat XYZ1, cv::Mat T, cv::Mat K )
{
    /*
    Arguments:
        XYZ:    Global postion of <MapPoint> [shape 4 x 1].
        T:      Camera global transformation matrix [shape 4 x 4].
        K:      Camera intrinsic paramters [shape 3 x 3].
    Returns:
        xy1:     Keypoint location in homogeneous pixel coordinates [shape 3 x 1].
    Overview:
        lambda * xy1 = K * I' * XYZ1
    */
    cv::Mat XYZ_camera_hom;// = cv::Mat::zeros(4, 1, CV_64F);
    cv::solve(T, XYZ1, XYZ_camera_hom);
    dehomogenizeMatrix(XYZ_camera_hom);
    cv::Mat reduce;
    reduce = (cv::Mat_<double>(3,4)<<1, 0, 0, 0,
                                     0, 1, 0, 0,
                                     0, 0, 1, 0);
    cv::Mat xy1 = K * reduce * XYZ_camera_hom;
    dehomogenizeMatrix(xy1);
    return xy1;
    //return cv::Mat::zeros(3, 1, CV_64F);
}

cv::Mat relTfromglobalTx2(cv::Mat T1, cv::Mat T2)
{
    /*
    Arguments:
        T1:     Global transformation matrix 1, newest
        T2:     Global transformation matrix 2, oldest
    Returns:
        rel_T:  Relative basis vector transformation between T1 and T2
    */
    cv::Mat rel_T = inverseTMatrix(T2) * T1;
    return rel_T;
}

std::vector<double> transform2stdParam(cv::Mat &T)
{
    cv::Mat R, t;
    R = cv::Mat::zeros(3,3,CV_64F);
    t = cv::Mat::zeros(3,1,CV_64F);
    T2RotAndTrans(T, R, t);
    std::vector<double> rot = rotationMatrixToEulerAngles(R);
    return std::vector<double> {rot[0], rot[1], rot[2], t.at<double>(0,0), t.at<double>(1,0), t.at<double>(2,0)};
}

bool isInsideImageBounds( double y, double x, int H, int W)
{
    if ( y >= H || y < 0 || x >= W || x < 0)
    {
        return false;
    }
    else
    {
        return true;
    }
}

void writeParameters2File(std::string file_path, std::string image_idenifier, cv::Mat &T )
{
    std::vector<double> p = transform2stdParam(T);

    std::ofstream pose_file;
    pose_file.open(file_path, std::ios_base::app);
    if (pose_file.is_open())
    {
        pose_file << image_idenifier << " " << p[0] << " " << p[1] << " " << p[2]
                                    << " " << p[3] << " " << p[4] << " " << p[5];
        pose_file << "\n";
        pose_file.close();
    }
    else
    {
        std::cout << "Unable to open file: " << file_path << std::endl;
    }
}

std::vector<std::vector<std::string>> readCSVFile(std::string filename)
{
    std::vector<std::vector<std::string>> content;
    std::vector<std::string> row;
	std::string line, word;
 
	std::fstream file (filename, std::ios::in);
	if(file.is_open())
	{
		while(getline(file, line))
		{
			row.clear();
 
			std::stringstream str(line);
 
			while(getline(str, word, ','))
            {
				row.push_back(word);
            }
            content.push_back(row);
		}
	}
    else
    {
        std::cout << "ERROR: COULD NOT OPEN FILE " << filename << std::endl;
    }
    return content;
}