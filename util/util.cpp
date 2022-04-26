#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
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

void drawIndicator(cv::Mat& img, double percentage, cv::Point pos)
{
    percentage = std::max(double(0), percentage);

    // config
    int levels = 10;
    int indicator_width = 80;
    int indicator_height = 180;
    int border_w = 20;
    int level_width = indicator_width - border_w;
    int level_height = int((indicator_height - border_w) / levels - 5);
    // draw
    int img_levels = int(percentage / levels);
    int top = pos.y;
    int left = pos.x;
    int bottom = pos.y + indicator_height;
    int right = pos.x + indicator_width;
    cv::rectangle(img, pos, cv::Point(right, bottom), (0, 0, 0), cv::FILLED);

    int level_y_b;
    for (int i = 0; i < levels; ++i)
    {
        if ( i <= img_levels )
        {
            level_y_b = int(bottom - (border_w + i * (level_height + 5)));
            cv::rectangle(img, cv::Point(left + border_w, level_y_b - level_height), cv::Point(left + border_w + level_width, level_y_b), percentage2Color(double(i) / double(levels)), cv::FILLED);
        }
    }
}
/*
void drawIndicator(cv::Mat& img, double percentage, cv::Point pos)
{
    // config
    int levels = 10;
    int indicator_width = 80;
    int indicator_height = 180;
    int level_width = indicator_width - 20;
    int level_height = int((indicator_height - 20) / levels - 5);
    // draw
    int img_levels = int(percentage * levels);
    cv::rectangle(img, cv::Point(10, img.rows - (indicator_height + 10)), cv::Point(10 + indicator_width, img.rows - 10), (0, 0, 0), cv::FILLED);

    int level_y_b;
    for (int i = 0; i < levels; ++i)
    {
        level_y_b = int(img.rows - (20 + i * (level_height + 5)));
        cv::rectangle(img, cv::Point(20, level_y_b - level_height), cv::Point(20 + level_width, level_y_b), percentage2Color(double(i) / double(levels)), cv::FILLED);
    }
}
*/

cv::Scalar percentage2Color(double p)
{
    cv::Scalar color(0, int(255 * p), int(255 - (255 * p)));
    return color;
}

cv::Mat composeRMatrix( double x_rot, double y_rot, double z_rot )
{

    // Calculate rotation about x axis
    cv::Mat R_x = (cv::Mat_<double>(3,3) <<
               1,               0,              0,
               0,               cos(x_rot),  -sin(x_rot),
               0,               sin(x_rot),  cos(x_rot)
               );
    
    // Calculate rotation about y axis
    cv::Mat R_y = (cv::Mat_<double>(3,3) <<
               cos(y_rot),   0,              sin(y_rot),
               0,               1,              0,
               -sin(y_rot),  0,              cos(y_rot)
               );
    
    // Calculate rotation about z axis
    cv::Mat R_z = (cv::Mat_<double>(3,3) <<
               cos(z_rot),   -sin(z_rot), 0,
               sin(z_rot),   cos(z_rot),  0,
               0,               0,              1
               );
    
    // Combined rotation matrix
    cv::Mat R = R_z * R_y * R_x;

    return R;
}

cv::Mat invertKMatrix( cv::Mat K )
{
    /*
    Arguments:
        K:  Camera matrix: [[fx, s,  cx]
                            [0,  fy, cy]
                            [0,  0,  1]]
    */
    //TODO: Implement inversion for other "configurations" of the K_matrix.
    double fx, fy, cx, cy, s;
    cv::Mat K_inv;

    fx = K.at<double>(0,0);
    fy = K.at<double>(1,1);
    cx = K.at<double>(0,2);
    cy = K.at<double>(1,2);
    s = K.at<double>(0,1);

    if ( fx == fy && s == 0 )
    {
        K_inv = (cv::Mat_<double>(3,3) << 1/fx,     0,      -cx/fx,
                                            0,      1/fx,   -cy/fx,
                                            0,      0,      1);
    }
    else if ( fx != fy && s == 0 )
    {
        K_inv = (cv::Mat_<double>(3,3) << 1/fx,     0,      -cx/fx,
                                            0,      1/fy,   -cy/fy,
                                            0,      0,      1);
    }
    else if ( fx == fy && s != 0 )
    {
        K_inv = (cv::Mat_<double>(3,3) << 1/fx,     -s/(fx*fx), (-fx*cx + s*cy)/(fx*fx),
                                            0,      1/fx,       -cy/fx,
                                            0,      0,          1);
    }
    else if ( fx != fy && s != 0 )
    {
        K_inv = (cv::Mat_<double>(3,3) << 1/fx,     -s/(fx*fy), (-fy*cx + s*cy)/(fx*fy),
                                            0,      1/fy,       -cy/fy,
                                            0,      0,          1);
    }
    else
    {
        std::cout << "ERROR: K^(-1) not defined for this configuration of K" << std::endl;
    }
    return K_inv;
}

cv::Mat composeEMatrix(cv::Mat& R, cv::Mat& t)
{
    /* Composes the E matrix from the R and t matrixes */
    cv::Mat t_skew =    (cv::Mat_<double>(3,3) <<
                            0,                  -t.at<double>(2,0),  t.at<double>(1,0),
                            t.at<double>(2,0),   0,                  -t.at<double>(0,0),
                            -t.at<double>(1,0),  t.at<double>(0,0),   0
                            );
    // Matrix multipliation
    cv::Mat E = t_skew * R;
    return E;
}

cv::Mat fundamentalFromEssential(cv::Mat E_matrix, cv::Mat K_matrix)
{
    /*
    Effect:
        Calculates the Fundamental matrix given the Essential matrix and a single camera matrix.
    */

    //cv::Mat K_inv = K_matrix.inv();
    cv::Mat K_inv = invertKMatrix(K_matrix);
    cv::Mat F = K_inv.t()*E_matrix*K_inv;
    return F;
}

cv::Mat fundamentalFromEssential(cv::Mat E_matrix, cv::Mat K1_matrix, cv::Mat K2_matrix)
{
    /*
    Effect:
        Calculates the Fundamental matrix given the Essential matrix and a two different camera matrices.
    */

    //cv::Mat K_inv = K_matrix.inv();
    cv::Mat K1_inv = invertKMatrix(K1_matrix);
    cv::Mat K2_inv = invertKMatrix(K2_matrix);
    cv::Mat F = K1_inv.t()*E_matrix*K2_inv;
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

void drawEpipolarLinesOld(cv::Mat F, cv::Mat &img_disp2,
                        std::vector<cv::Point2f> points1,
                        std::vector<cv::Point2f> points2)
{
    std::vector<cv::Vec<double,3>> epilines1;
    std::cout << "111111" << std::endl;
    cv::computeCorrespondEpilines(points1, 1, F, epilines1); //Index starts with 1
    std::cout << "!!!!!!" << std::endl;

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

void drawEpipolarLines(cv::Mat F, cv::Mat &img_disp2,
                        std::vector<cv::Point2f> points1,
                        std::vector<cv::Point2f> points2)
{
    //std::vector<cv::Vec<double,3>> epilines1;
    std::vector<cv::Point3f> epilines1;
    cv::computeCorrespondEpilines(points1, 1, F, epilines1); //Index starts with 1

    cv::RNG rng(0);
    for(size_t i=0; i<points1.size(); i++)
    {
        /*
        * Epipolar lines of the 1st point set are drawn in the 2nd image and vice-versa
        */
        cv::Scalar color(rng.uniform(0,256),rng.uniform(0,256),rng.uniform(0,256));
        cv::line(img_disp2,
            cv::Point(0,-epilines1[i].z/epilines1[i].y),
            cv::Point(img_disp2.cols,-(epilines1[i].z+epilines1[i].x*img_disp2.cols)/epilines1[i].y),
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

cv::Mat homogenizeArrayRet(const cv::Mat& arr)
{
    cv::Mat ret = arr.clone();
    cv::Mat row = cv::Mat::ones(1, arr.cols, CV_64F);
    ret.push_back(row);
    return ret;
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

cv::Mat normalizeMat(cv::Mat& vec)
{
    cv::Mat ret;
    std::cout << "Input: " << vec << std::endl;
    double norm = cv::norm(vec, cv::NORM_L1);
    ret = vec/norm;
    std::cout << "Output: " << ret << std::endl;
    std::cout << "Norm: " << norm << std::endl;
    return ret;
}

cv::Mat fitQuadraticForm(cv::Mat& x, cv::Mat& y, cv::Mat& z)
{

    /*
    Arguments:
        x:      List of x coordinates,  [N x 1].
        y:      List of y coordinates,  [N x 1].
        z:      List of z coordinates,  [N x 1].
    Returns:
        A:      Quadratic form fitted to (y, x, 1).T * A * (y, x, 1) = z, [3 x 3].
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
        D.at<double>(i, 0) = y_i*y_i;
        D.at<double>(i, 1) = x_i*x_i;
        D.at<double>(i, 2) = x_i*y_i;
        D.at<double>(i, 3) = y_i;
        D.at<double>(i, 4) = x_i;
    }

    cv::Mat a;
    cv::solve(D, z, a, cv::DECOMP_SVD);
    cv::Mat A = (cv::Mat_<double>(3,3)<<a.at<double>(0,0),      a.at<double>(0,2)/2,    a.at<double>(0,3)/2,
                                    a.at<double>(0,2)/2,    a.at<double>(0,1),      a.at<double>(0,4)/2,
                                    a.at<double>(0,3)/2,    a.at<double>(0,4)/2,    a.at<double>(0,5));

    /*
    double error = cv::norm(z-D*a);
    std::cout << "Error of the least-squares solution: ||b-A*x|| = " << error << std::endl;
    //std::cout << D*a << std::endl;
    //std::cout << z << std::endl;
    cv::Mat err;
    cv::Mat err_mat;
    for (int i = 0; i < x.rows; i++)
    {
        cv::Mat Y = (cv::Mat_<double>(3,1)<<y.at<double>(i,0),
                                            x.at<double>(i,0),
                                            1);
        double b = z.at<double>(i,0);
        err_mat = Y.t() * A * Y - b;
        err.push_back(err_mat.at<double>(0,0));
    }
    std::cout << "Calc error: " << cv::norm(err) << std::endl;
    */
    return A;
}

cv::Mat reprojectionError( cv::Mat& xyz1, cv::Mat& uv1, cv::Mat& T, cv::Mat& K )
{
    /*
    Arguments:
        xyz1:   3D location of keypoint, dehomogenized [4 x 1].
        uv1:    Keypoint location in homogeneous pixel coordinates [shape 3 x 1].
        T:      Camera global transformation matrix [shape 4 x 4].
        K:      Camera intrinsic paramters [shape 3 x 3].
    Returns:
        uv_err: Reprojection error in individual directions, homogeneous coordinates [3 x 1].
    */
    cv::Mat uv1_proj, uv1_err;

    uv1_proj = projectKpt( xyz1, T, K );
    uv1_err = uv1_proj - uv1;
    return uv1_err;
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

void triangulatePointsLinear( cv::Mat& rel_T, cv::Mat& K1, cv::Mat& K2, cv::Mat& uv1, cv::Mat& uv2, cv::Mat& XYZ_I2 )
{
    /*
    Arguments:
        rel_T:      Relative transformation from image 2 to image 1 [3 x 4].
        K1:         Intrinsic camera matrix for image 1 [3 x 3].
        K2:         Intrinsic camera matrix for image 2 [3 x 3].
        uv1:        Non-normalized coordinates of keypoints in image 1 [2 x N]/[3 x N].
        uv2:        Non-normalized coordinates of keypoints in image 2 [2 x N]/[3 x N].
    Returns:
        XYZ_I2:     Triangulated points in 3D coordinates with respect to image 2, homogenized [3xN].
    */
    cv::Mat_<double> coord_3D;

    uv1 = uv1.rowRange(0,2);
    uv2 = uv2.rowRange(0,2);

    double fx1 = K1.at<double>(0,0);
    double fy1 = K1.at<double>(1,1);
    double cx1 = K1.at<double>(0,2);
    double cy1 = K1.at<double>(1,2);

    double fx2 = K2.at<double>(0,0);
    double fy2 = K2.at<double>(1,1);
    double cx2 = K2.at<double>(0,2);
    double cy2 = K2.at<double>(1,2);

    uv1.row(0) = (uv1.row(0) - cx1) / fx1;
    uv2.row(0) = (uv2.row(0) - cx2) / fx2;
    uv1.row(1) = (uv1.row(1) - cy1) / fy1;
    uv2.row(1) = (uv2.row(1) - cy2) / fy2;

    cv::triangulatePoints(cv::Mat::eye(3,4, CV_64F), rel_T, uv1, uv2, coord_3D);
    XYZ_I2 = coord_3D;
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

std::string type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
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