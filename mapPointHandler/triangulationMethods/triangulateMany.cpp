
#include "triangulateMany.hpp"
#include "../../util/util.hpp"

cv::Mat linearTriangulateMany(cv::Mat& uv1, cv::Mat& uv2, cv::Mat& K1, cv::Mat& K2, cv::Mat& T1, cv::Mat& T2)
{
    /*  Arguments:
            uv:     Homogeneous pixel coordinates in image 1 and 2 [shape 3 x n].
            K:      Kalibration matrix for image 1 and 2 [shape 3 x 3].
            T:      Global extrinsic matrix of 1 and 2 [shape 4 x 4].
        Returns:
            XYZ:    Dehomogenized 3D point in the world frame [shape 4 x n].
    */

    int n = uv1.size[1];
    cv::Mat xy1, xy2, P1, P2, A, S, U, VT, XYZ;

    xy1 = K1.inv() * uv1;
    xy2 = K2.inv() * uv2;
    P1 = T1.rowRange(0,3).colRange(0,4);
    P2 = T2.rowRange(0,3).colRange(0,4);

    XYZ = cv::Mat(4, n, CV_64F);
    #pragma omp parallel for
    for (int i = 0; i < n; i++)
    {   
        A = cv::Mat::zeros(4,4,CV_64F);
        A.row(0) = P1.row(0) - xy1.at<double>(0,i)*P1.row(2);
        A.row(1) = P1.row(1) - xy1.at<double>(1,i)*P1.row(2);
        A.row(2) = P2.row(0) - xy2.at<double>(0,i)*P2.row(2);
        A.row(3) = P2.row(1) - xy2.at<double>(1,i)*P2.row(2);
        cv::SVDecomp(A, S, U, VT, cv::SVD::FULL_UV);
        XYZ.col(i) = (VT.row(3)/VT.at<double>(3,3)).t();
    }
    return XYZ;
}

int opencvTriangulationTest(cv::InputArray _points1, cv::InputArray _points2, cv::InputArray _cameraMatrix, cv::InputArray _P1, cv::InputArray _P2, cv::OutputArray triangulatedPoints)
{
    // Assumes only one camera matrix for both views
    
    cv::Mat points1, points2, cameraMatrix, P1, P2;
    _points1.getMat().convertTo(points1, CV_64F);
    _points2.getMat().convertTo(points2, CV_64F);
    _P1.getMat().convertTo(P1, CV_64F);
    _P2.getMat().convertTo(P2, CV_64F);
    _cameraMatrix.getMat().convertTo(cameraMatrix, CV_64F);

    int npoints = points1.checkVector(2);
    CV_Assert( npoints >= 0 && points2.checkVector(2) == npoints &&
                              points1.type() == points2.type());

    CV_Assert(cameraMatrix.rows == 3 && cameraMatrix.cols == 3 && cameraMatrix.channels() == 1);

    if (points1.channels() > 1)
    {
        points1 = points1.reshape(1, npoints);
        points2 = points2.reshape(1, npoints);
    }

    double fx = cameraMatrix.at<double>(0,0);
    double fy = cameraMatrix.at<double>(1,1);
    double cx = cameraMatrix.at<double>(0,2);
    double cy = cameraMatrix.at<double>(1,2);

    points1.col(0) = (points1.col(0) - cx) / fx;
    points2.col(0) = (points2.col(0) - cx) / fx;
    points1.col(1) = (points1.col(1) - cy) / fy;
    points2.col(1) = (points2.col(1) - cy) / fy;

    points1 = points1.t();
    points2 = points2.t();

    cv::Mat Q;

    triangulatePoints(P1, P2, points1, points2, Q);
    Q.copyTo(triangulatedPoints);

    cv::Mat mask1 = Q.row(2).mul(Q.row(3)) > 0;
    Q.row(0) /= Q.row(3);
    Q.row(1) /= Q.row(3);
    Q.row(2) /= Q.row(3);
    Q.row(3) /= Q.row(3);
    Q = P2 * Q;
    mask1 = (Q.row(2) > 0) & mask1;

    mask1 = mask1.t();

    int good1 = countNonZero(mask1);

    return good1;
}

void triangulateTest(cv::Mat K1, cv::Mat K2)
{
    // TRIANGULATE TEST
    cv::Mat X(4,3, CV_64F);
    cv::randu(X, cv::Scalar(-5), cv::Scalar(5));
    X.row(3) = cv::Mat::ones(1,3, CV_64F);
    cv::Mat R1 = (cv::Mat_<double>(3,4)<<   0.9211, 0.0000,-0.3894, 0.0000,
                                            0.0000, 1.0000, 0.0000, 0.0000,
                                            0.3894, 0.0000, 0.9211, 6.0000);
    cv::Mat R2 = (cv::Mat_<double>(3,4)<<   0.9211, 0.0000, 0.3894, 0.0000,
                                            0.0000, 1.0000, 0.0000, 0.0000,
                                            -0.3894, 0.0000, 0.9211, 6.0000);
    cv::Mat x1,x2, X_hat;
    R1 = K1 * R1;
    R2 = K2 * R2;
    x1 = R1*X;
    x2 = R2*X;
    for(int i =0; i < x1.cols; i++ )
    {
        x1.at<double>(0, i) /= x1.at<double>(2, i);
        x1.at<double>(1, i) /= x1.at<double>(2, i);
        x1.at<double>(2, i) /= x1.at<double>(2, i);
        x2.at<double>(0, i) /= x2.at<double>(2, i);
        x2.at<double>(1, i) /= x2.at<double>(2, i);
        x2.at<double>(2, i) /= x2.at<double>(2, i);
    }
    x1 = x1.rowRange(0,2);
    x2 = x2.rowRange(0,2);
    cv::triangulatePoints(R1, R2, x1, x2, X_hat);
    dehomogenizeMatrix( X_hat );
    std::cout <<"TRIANGULATE TEST: " << std::endl;
    std::cout << x1 << std::endl;
    std::cout << x2 << std::endl;
    std::cout << X << std::endl;
    std::cout << X_hat << std::endl;

}
