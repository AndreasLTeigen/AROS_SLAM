#include <memory>
#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>

#include "GJET.hpp"
#include "../../util/util.hpp"

using std::vector;
using std::shared_ptr;

// Epipolar Constrained Optimization Solver
struct ECOptSolver 
{
    ECOptSolver(const cv::Mat x_k, const cv::Mat y_k, const cv::Mat A_k, const cv::Mat K1, const cv::Mat K2, 
                            const shared_ptr<Parametrization> parametrization)
                {
                    x_k_ = x_k;
                    y_k_ = y_k;
                    A_k_ = A_k;
                    K1_ = K1;
                    K2_ = K2;
                    parametrization_ = parametrization;
                }
                    //: x_k(x_k), y_k(y_k), K1(K1), K2(K2), parametrization(parametrization) {}
    
    bool operator()( const double* p, double* residual ) const
    {
        cv::Mat R, t, E_matrix, F_matrix, v_k_opt;

        vector<double> p_vec;
        for ( int i = 0; i < 6; ++i )
        {
            p_vec.push_back(p[i]);
        }
        /*
        for ( double val : p_vec )
        {
            std::cout << val << ", ";
        }
        std::cout << "\n";
        */
        
        parametrization_->composeRMatrixAndTParam( p_vec, R, t );
        E_matrix = composeEMatrix( R, t );
        F_matrix = fundamentalFromEssential( E_matrix, K1_, K2_ );
        residual[0] = GJET::epipolarConstrainedOptimization( F_matrix, A_k_, x_k_, y_k_, v_k_opt );
        return true;
    }


    cv::Mat K1_, K2_, x_k_, y_k_, A_k_;
    shared_ptr<Parametrization> parametrization_;
};

struct GJETSolver
{
    GJETSolver(const cv::Mat x_k, const cv::Mat y_k, const cv::Mat A_k, const cv::Mat K1, const cv::Mat K2, 
                        const shared_ptr<Parametrization> parametrization)
                    : ecopt_solver(new ceres::NumericDiffCostFunction<ECOptSolver, ceres::CENTRAL, 1, 6>(
                                                new ECOptSolver(x_k, y_k, A_k, K1, K2, parametrization))) {}
    
    template <typename T>
    bool operator()(const T* p, T* residual) const
    {
        return ecopt_solver(p, residual);
    }

    private:
        ceres::CostFunctionToFunctor<1,6> ecopt_solver;
};



std::shared_ptr<Pose> GJET::calculate( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2 )
{
    // Assumes K_matrix is equal for both frames.
    cv::Mat E_matrix, F_matrix, inliers;
    std::vector<cv::Point> pts1, pts2;
    compileMatchedCVPoints( frame1, frame2, pts1, pts2 );
    E_matrix = cv::findEssentialMat( pts1, pts2, frame1->getKMatrix(), cv::RANSAC, 0.999, 1.0, inliers );
    FrameData::removeOutlierMatches( inliers, frame1, frame2 );

    vector<shared_ptr<KeyPoint2>> matched_kpts1 = frame1->getMatchedKeypoints( frame2->getFrameNr() );
    vector<shared_ptr<KeyPoint2>> matched_kpts2 = frame2->getMatchedKeypoints( frame1->getFrameNr() );

    std::shared_ptr<Pose> rel_pose = FrameData::registerRelPose( E_matrix, frame1, frame2 );
    rel_pose->updateParametrization();

    // ------ CERES test -------------
    ceres::Problem problem;
    int N = matched_kpts1.size();
    cv::Mat A_d_k, x_k, y_k;
    shared_ptr<KeyPoint2> kpt1, kpt2;
    shared_ptr<StdParam> parametrization = std::make_shared<StdParam>();
    vector<double> p_init = rel_pose->getParametrization( this->paramId )->getParamVector();
    std::cout << *rel_pose->getParametrization() << "\n";
    double p[p_init.size()];// = p_init.data();

    F_matrix = fundamentalFromEssential( E_matrix, frame1->getKMatrix(), frame2->getKMatrix() );
    this->jointEpipolarOptimization( F_matrix, matched_kpts1, matched_kpts2 );
    
    //Filling p with values from p_init
    for ( int i = 0; i < p_init.size(); ++i )
    {
        p[i] = p_init[i];
    }

    for ( int i = 0; i < p_init.size(); ++i )
    {
        std::cout << p[i] << ", ";
    }

    std::cout << "\n";
    std::cout << sizeof(p)/sizeof(p[0]) << std::endl;

    
    for ( int n = 0; n < N; ++n )
    {
        cv::Mat v_k;
        kpt1 = matched_kpts1[n];
        kpt2 = matched_kpts2[n];
        y_k = kpt1->getLoc();
        x_k = kpt2->getLoc();
        A_d_k = kpt1->getDescriptor("quad_fit");

        ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<GJETSolver, 1, 6>(
            new GJETSolver(x_k, y_k, A_d_k, frame1->getKMatrix(), frame2->getKMatrix(), parametrization));
        problem.AddResidualBlock(cost_function, nullptr, p);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 100;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";

    std::cout << "p: ";
    for (int i = 0; i < 6; i++)
    {
        std::cout << p_init[i] << ", ";
    }
    std::cout << "\n ->";
    for (int i = 0; i < 6; i++)
    {
        std::cout << p[i] << ", ";
    }
    std::cout << "\n";
    
    vector<double> p_vec;
    for ( int i = 0; i < 6; ++i )
    {
        p_vec.push_back(p[i]);
    }
    rel_pose->setPose( p_vec, this->paramId );
    std::cout << *rel_pose->getParametrization() << "\n";
    std::cout << rel_pose->getTMatrix() << std::endl;
    F_matrix = fundamentalFromEssential( rel_pose->getEMatrix(), frame1->getKMatrix(), frame2->getKMatrix() );
    this->jointEpipolarOptimization( F_matrix, matched_kpts1, matched_kpts2 );
    //--------------------------------------

    return rel_pose;
}

double GJET::solveQuadraticFormForV( cv::Mat& A_k, cv::Mat& b_k, cv::Mat& c_k, cv::Mat& v_k )
{
    /*
    Effect: Calculates the quadratic form for y_k + v_k:    
                v_k.T * A_k * v_k + 2 * v_k.T * b_k + c_k
            where:
                A_k = eye(2,3) * A_d_k * eye(2,3).T
                b_k = eye(2,3) * A_d_k * (y_k, 1).T
                c_k = (y_k.T, 1) * A_d_k * (y_k, 1).T
    */
    cv::Mat g = v_k.t() * A_k * v_k + 2 * v_k.t() * b_k + c_k;
    return g.at<double>(0,0);
}

cv::Mat GJET::solveKKT( cv::Mat& A, cv::Mat& g, cv::Mat& b, cv::Mat& h )
{
    /*
    Effect:
        Solves the KKT matrix with a 2D vector optimization:
                [[A,    g],   *     [x1, x2, lambda].t()    =   [b, h]
                 [g.t(),0]]
        where:
                A = [[a11, a12],    g = [g1 ,g2].t(),   b = [b1, b2].t()
                     [a12, a22]]
            and x1, x2, lambda and h are scalars.
    */

    double a11, a12, a22, g1, g2, b1, b2, h1, x1, x2;
    a11 =   A.at<double>(0,0);
    a12 =   A.at<double>(0,1);
    a22 =   A.at<double>(1,1);
    g1 =    g.at<double>(0,0);
    g2 =    g.at<double>(1,0);
    b1 =    b.at<double>(0,0);
    b2 =    b.at<double>(1,0);
    h1 =    h.at<double>(0,0);

    x1 = (g1*g2*b2 - a22*g1*h1 - b1*g2*g2 + a12*g2*h1)/(2*a12*g1*g2 - a22*g1*g1 - a11*g2*g2);
    x2 = (h1 - g1*x1)/g2;
    
    cv::Mat x = (cv::Mat_<double>(2,1)<< x1,
                                         x2);
    return x;
}

double GJET::epipolarConstrainedOptimization(const cv::Mat& F_matrix, const cv::Mat& A_d_k, const cv::Mat& x_k, const cv::Mat& y_k, cv::Mat& v_k_opt )
{
    /*
    Arguements: 
        A_d_k:      Image information based loss function in quadratic form (y_k.T*A_d_k*y_k) for keypoint 'k' [3 x 3].
        F_matrix:   Fundamental matrix between image 1 and image 2 [3 x 3].
        y_k:        Location of keypoint k in image 1, corresponding to y_k [2 x 1].
        x_k:        Location of keypoint k in image 2, corresponding to x_k [2 x 1].
    Returns:
        v_k_opt:    Perturbation off of y_k in image 2, optimized for A_k and constrained by the epipolar line [2 x 1].
        ret:        Value of 'y_k + v_k_opt' on the quadratic function.
    */
    cv::Mat A_k, b_k, c_k, F_d, F_d_x, KKT, q, q_31, b_k_neg, y1_k, x1_k;
    cv::Mat I_s = cv::Mat::eye(2, 3, CV_64F);

    y1_k = homogenizeArrayRet(y_k);
    x1_k = homogenizeArrayRet(x_k);

    // Helping definitions
    F_d = I_s * F_matrix;
    F_d_x = F_d*x1_k;
    q_31 = -(y1_k.t()) * F_matrix * (x1_k);
    
    // KKT sub matrixes/vectors
    A_k = I_s * A_d_k * I_s.t();
    b_k = I_s * A_d_k * y1_k;
    c_k = y1_k.t() * A_d_k * y1_k;


    /*
    KKT = (cv::Mat_<double>(3,3)<<  A_k.at<double>(0,0),    A_k.at<double>(0,1),    F_d_x.at<double>(0,0),
                                    A_k.at<double>(1,0),    A_k.at<double>(1,1),    F_d_x.at<double>(1,0),
                                    F_d_x.at<double>(0,0),  F_d_x.at<double>(1.0),  0);


    q = (cv::Mat_<double>(3,1)<<    -b_k.at<double>(0,0),
                                    -b_k.at<double>(1,0),
                                     q_31.at<double>(0,0));
    */

    //SOLVE KKT Problem KKT*x = q!
    b_k_neg = -b_k;
    v_k_opt = GJET::solveKKT( A_k, F_d_x, b_k_neg, q_31 );


    return GJET::solveQuadraticFormForV( A_k, b_k, c_k, v_k_opt );
}

void GJET::jointEpipolarOptimization( cv::Mat& F_matrix, vector<shared_ptr<KeyPoint2>>& matched_kpts1, vector<shared_ptr<KeyPoint2>>& matched_kpts2 )
{
    /*
    Arguments:
        F_matrix:       Fundamental matrix between image 1 and image 2 [3 x 3].
        matched_kpts1:  List of matched keypoints from image 1 with image 2 [N].
        matched_kpts2:  List of matched keypoints from image 2 with image 2 [N].
    Returns:
        v_opt:          Perturbations of all kepoints from their detected positions that are in-lign with optimized 
                        epipolar geometry and the loss function based on the image information [2 x N].
    */
    
    int N = matched_kpts1.size();
    double loss_n;
    cv::Mat A_d_k, x_k, y_k;
    shared_ptr<KeyPoint2> kpt1, kpt2;

    double tot_loss = 0;
    
    
    for ( int n = 0; n < N; ++n )
    {
        cv::Mat v_k;// = cv::Mat::zeros(2,1,CV_64F);
        kpt1 = matched_kpts1[n];
        kpt2 = matched_kpts2[n];
        y_k = kpt1->getLoc();
        x_k = kpt2->getLoc();
        A_d_k = kpt1->getDescriptor("quad_fit");

        loss_n = GJET::epipolarConstrainedOptimization( F_matrix, A_d_k, x_k, y_k, v_k );

        tot_loss += loss_n*loss_n;
        
        /*
        std::cout << "---------------------" << std::endl;
        std::cout << "Kpt nr: " << n << std::endl;
        std::cout << homogenizeArrayRet(y_k).t() * F_matrix * homogenizeArrayRet(x_k) << std::endl;
        std::cout << "Kpt\n" << y_k << std::endl;
        std::cout << "v_k: \n" << v_k << std::endl;
        std::cout << "Loss: " << loss_n << std::endl;
        std::cout << "A:\n" << A_d_k << std::endl;
        */
    }
    std::cout << "Total Loss: " << tot_loss << std::endl;
}

