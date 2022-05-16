#include <cmath>
#include <memory>
#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>

#include "GJET.hpp"
#include "../../util/util.hpp"

using std::vector;
using std::shared_ptr;
using cv::Mat;

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::high_resolution_clock;


// Epipolar Constrained Optimization Solver
struct ECOptSolver 
{
    ECOptSolver( const cv::Mat K1, const cv::Mat K2, const shared_ptr<KeyPoint2> kpt1, 
                        const shared_ptr<KeyPoint2> kpt2, cv::Mat& img, const shared_ptr<Parametrization> parametrization, const shared_ptr<LossFunction> paraboloidNormal )
                {
                    K1_ = K1;
                    K2_ = K2;
                    img_ = img;
                    kpt1_ = kpt1;
                    kpt2_ = kpt2;
                    parametrization_ = parametrization;
                    paraboloidNormal_ = paraboloidNormal;
                }
    
    bool operator()( const double* p, double* residual ) const
    {
        cv::Mat R, t, y_k, x_k, E_matrix, F_matrix, v_k_opt;

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
        
        y_k = kpt1_->getLoc();
        x_k = kpt2_->getLoc();
        
        parametrization_->composeRMatrixAndTParam( p_vec, R, t );
        E_matrix = composeEMatrix( R, t );
        F_matrix = fundamentalFromEssential( E_matrix, K1_, K2_ );
        // TODO:: REVERT THIS
        residual[0] = GJET::epipolarConstrainedOptimization( F_matrix, kpt1_->getDescriptor("quad_fit"), x_k, y_k, v_k_opt );
        //residual[0] = GJET::reprojectionError(F_matrix, x_k, y_k, v_k_opt);
        return true;
    }


    cv::Mat K1_, K2_, A_k_, img_;
    shared_ptr<KeyPoint2> kpt1_, kpt2_;
    shared_ptr<Parametrization> parametrization_;
    shared_ptr<LossFunction> paraboloidNormal_;
};

struct GJETSolver
{
    GJETSolver(const cv::Mat K1, const cv::Mat K2, const shared_ptr<KeyPoint2> kpt1, 
                        const shared_ptr<KeyPoint2> kpt2, cv::Mat& img, const shared_ptr<Parametrization> parametrization, const shared_ptr<LossFunction> paraboloidNormal)
                    : ecopt_solver(new ceres::NumericDiffCostFunction<ECOptSolver, ceres::CENTRAL, 1, 6>(
                                                new ECOptSolver(K1, K2, kpt1, kpt2, img, parametrization, paraboloidNormal))) {}
    
    template <typename T>
    bool operator()(const T* p, T* residual) const
    {
        return ecopt_solver(p, residual);
    }

    private:
        ceres::CostFunctionToFunctor<1,6> ecopt_solver;
};
















std::shared_ptr<Pose> GJET::calculate( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2, cv::Mat& img )
{
    // Assumes K_matrix is equal for both frames.
    cv::Mat E_matrix, F_matrix, inliers;
    std::vector<cv::Point> pts1, pts2;
    compileMatchedCVPoints( frame1, frame2, pts1, pts2 );
    E_matrix = cv::findEssentialMat( pts1, pts2, frame1->getKMatrix(), cv::RANSAC, 0.999, 1.0, inliers );
    FrameData::removeOutlierMatches( inliers, frame1, frame2 );
    std::shared_ptr<Pose> rel_pose = FrameData::registerRelPose( E_matrix, frame1, frame2 );
    rel_pose->updateParametrization();

    shared_ptr<LossFunction> loss_func = std::make_shared<LossFunction>();

    vector<shared_ptr<KeyPoint2>> matched_kpts1 = frame1->getMatchedKeypoints( frame2->getFrameNr() );
    vector<shared_ptr<KeyPoint2>> matched_kpts2 = frame2->getMatchedKeypoints( frame1->getFrameNr() );


    loss_func->computeParaboloidNormalForAll( matched_kpts1, matched_kpts2, img );
    //F_matrix = fundamentalFromEssential( E_matrix, frame1->getKMatrix(), frame2->getKMatrix() );
    //this->jointEpipolarOptimization( F_matrix, matched_kpts1, matched_kpts2 );

    
    // ------ CERES test -------------
    int N = matched_kpts1.size();
    cv::Mat A_d_k, x_k, y_k;
    shared_ptr<KeyPoint2> kpt1, kpt2;

    // Initializing parameter vector
    vector<double> p_init = rel_pose->getParametrization( this->paramId )->getParamVector();
    std::cout << *rel_pose->getParametrization( this->paramId ) << "\n";
    double p[p_init.size()];// = p_init.data();
    
    //Filling p with values from p_init
    for ( int i = 0; i < p_init.size(); ++i )
    {
        p[i] = p_init[i];
    }

    shared_ptr<StdParam> parametrization = std::make_shared<StdParam>();

    KeyPointUpdate itUpdate( img, p, frame1->getKMatrix(), frame2->getKMatrix(), loss_func, parametrization );  
    ceres::Problem::Options problem_options;
    problem_options.evaluation_callback = &itUpdate; 
    ceres::Problem problem(problem_options);

    for ( int i = 0; i < p_init.size(); ++i )
    {
        std::cout << p[i] << ", ";
    }

    std::cout << "\n";
    std::cout << sizeof(p)/sizeof(p[0]) << std::endl;
    
    for ( int n = 0; n < N; ++n )
    {
        kpt1 = matched_kpts1[n];
        A_d_k = kpt1->getDescriptor("quad_fit");
        if (!A_d_k.empty())
        {
            kpt2 = matched_kpts2[n];
            ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<GJETSolver, 1, 6>(
                new GJETSolver(frame1->getKMatrix(), frame2->getKMatrix(), kpt1, kpt2, img, parametrization, loss_func));
            //ceres::CostFunction* cost_function = new ceres::NumericDiffCostFunction<GJETSolver, ceres::CENTRAL, 1, 6>(
            //    new GJETSolver(frame1->getKMatrix(), frame2->getKMatrix(), kpt1, kpt2, img, parametrization, paraboloidNormal));
            problem.AddResidualBlock(cost_function, nullptr, p);

            itUpdate.addEvalKpt(kpt1, kpt2);
        }
    }             

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 50;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";

    itUpdate.moveKptToOptLoc();

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
    cv::Mat R_matrix = rel_pose->getRMatrix();
    cv::Mat t_vector = rel_pose->gettvector();
    t_vector = normalizeMat(t_vector);
    rel_pose->updatePoseVariables( R_matrix, t_vector );
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
                [[A,    g],   *     [x1, x2, lambda].t()    =   [b, h].t()
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
    cv::Mat A_k, b_k, c_k, F_d, F_d_x, KKT, q_31, b_k_neg, y1_k, x1_k;
    cv::Mat I_s = cv::Mat::eye(2, 3, CV_64F);

    //y1_k = homogenizeArrayRet(y_k);
    //x1_k = homogenizeArrayRet(x_k);
    y1_k = y_k;
    x1_k = x_k;

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
        kpt1 = matched_kpts1[n];
        A_d_k = kpt1->getDescriptor("quad_fit");
        if (!A_d_k.empty())
        {
            cv::Mat v_k;// = cv::Mat::zeros(2,1,CV_64F);
            kpt2 = matched_kpts2[n];
            y_k = kpt1->getLoc();
            x_k = kpt2->getLoc();

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
    }
    std::cout << "Total Loss: " << tot_loss << std::endl;
}


// Comparison function

double GJET::reprojectionError(const cv::Mat& F_matrix, const cv::Mat& x_k, const cv::Mat& y_k, cv::Mat& v_k_opt)
{
    double a, b, c, x0, y0, epi_x, epi_y, loss;
    cv::Mat epiline;
    epiline = F_matrix * x_k;
    a = epiline.at<double>(0);
    b = epiline.at<double>(1);
    c = epiline.at<double>(2);
    x0 = y_k.at<double>(0,0);
    y0 = y_k.at<double>(1,0);

    epi_x = (b*(b*x0 - a*y0) - a*c)/(a*a + b*b);
    epi_y = (a*(-b*x0 + a*y0) - b*c)/(a*a + b*b);

    cv::Mat y_k_opt = (cv::Mat_<double>(3,1) << epi_x, epi_y, 1);
    v_k_opt = y_k_opt - y_k;
    v_k_opt.at<double>(2,0) = 1;

    loss = cv::norm(v_k_opt);

    return loss*loss;
}

























// ################ Collecting descriptor differences ####################
LossFunction::LossFunction( cv::Mat img )
{
    this->W = img.cols;
    this->H = img.rows;
    this->img = img;
}

LossFunction::~LossFunction(){}

void LossFunction::collectDescriptorDistance( const cv::Mat& img, shared_ptr<KeyPoint2> kpt1, shared_ptr<KeyPoint2> kpt2 )
{
    cv::Mat local_descs, target_desc, A, hamming_dists, x, y, z;
    vector<cv::KeyPoint> local_kpts;

    local_kpts = this->generateLocalKpts( kpt1, img );
    this->orb->compute( img, local_kpts, local_descs );

    //target_desc = kpt1->getDescriptor(orb_non_rot);
    target_desc = kpt2->getDescriptor(this->orb_non_rot);
    hamming_dists = this->computeHammingDistance(target_desc, local_descs);
    this->generateCoordinateVectors(kpt1->getCoordX(), kpt1->getCoordY(), this->reg_size, x, y);
    z = hamming_dists.t();

    //std::cout << "[" << int(kpt1->getCoordY()) << ", " << int(kpt1->getCoordX()) << "]\n";
    //this->printKptLoc( local_kpts, this->reg_size, this->reg_size );
    //this->printLocalHammingDists( z, this->reg_size );

    A = fitQuadraticForm(x, y, z);
    kpt1->setDescriptor( A, "quad_fit" );
    kpt1->setDescriptor( z.reshape(1, this->reg_size), "hamming");
}

vector<cv::KeyPoint> LossFunction::generateLocalKpts( shared_ptr<KeyPoint2> kpt, const cv::Mat& img )
{

    int W, H, desc_radius;
    double ref_x, ref_y, x, y, kpt_x, kpt_y, kpt_size;
    vector<int> removal_kpts;
    vector<cv::KeyPoint> local_kpts;
    W = img.cols;
    H = img.rows;

    kpt_x = kpt->getCoordX();
    kpt_y = kpt->getCoordY();
    kpt_size = kpt->getSize();

    // Skips keypoints if local region will not produce all valid descriptors.
    desc_radius = std::max(this->patchSize, int(std::ceil(kpt_size)/2));
    if ( !validDescriptorRegion(kpt_x, kpt_y, W, H, desc_radius + this->reg_size) )
    {
        return local_kpts;
    }
    else
    {
        ref_x = kpt_x - reg_size/2; 
        ref_y = kpt_y - reg_size/2;
        for ( int row_i = 0; row_i < reg_size; ++row_i )
        {
            y = ref_y + row_i;
            for ( int col_j = 0; col_j < reg_size; ++col_j )
            {
                x = ref_x + col_j;
                local_kpts.push_back(cv::KeyPoint(x,y,kpt_size));
            }
        }
    }
    
    return local_kpts;
}

void LossFunction::printKptLoc( vector<cv::KeyPoint> kpts, int rows, int cols )
{
    cv::KeyPoint kpt;

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            kpt = kpts[i*rows + j];
            std::cout << "(" << std::round(kpt.pt.y) << ", " << std::round(kpt.pt.x) << ") , ";
        }
        std::cout << "\n";
    }
    std::cout << "----------------------" << std::endl;
}

void LossFunction::printLocalHammingDists( cv::Mat& hamming_dist_arr, int s )
{
    for (int row = 0; row < s; row++){
        for (int col = 0; col < s; col++)
        {
            std::cout << hamming_dist_arr.at<double>(row*s + col, 0) << ", ";
        }
        std::cout << "\n";
    }
    std::cout << "--------------------" << std::endl;
}

Mat LossFunction::computeHammingDistance( Mat& target_desc, Mat& region_descs )
{
    /*
    Arguments:
        target_desc:        Descriptor all other descriptors should be calculated the distance to.
        descs:              All other descriptors.
        N:                  Number of descriptors.
    Returns:
        desc_dists:         Hamming distance between <target_desc> and all descriptors in <descs>
    */

    int N = region_descs.rows;
    Mat hamming_dists = Mat::zeros(1, N, CV_64F);
    for ( int i = 0; i < N; ++i )
    {
        hamming_dists.at<double>(0, i) = cv::norm(target_desc, region_descs.row(i), cv::NORM_HAMMING);
    }
    return hamming_dists;
}

void LossFunction::generateCoordinateVectors(double x_c, double y_c, int size, Mat& x, Mat& y)
{
    /*
    Argument:
        x_c:    x coordinate of region center.
        y_c:    y coordinate of region center.
        size:   Size of the region (length of side).
    
    Returns:
        x:      x vector of coordinates that constitutes region, [size*size, 1].
        y:      y vector of coordinates that constitutes region, [size*size, 1].
    */

    // ref_x and ref_y are the top left coordinates of the region.
    int ref_x, ref_y, idx;
    Mat x_ret(size*size, 1, CV_64F);
    Mat y_ret(size*size, 1, CV_64F);

    ref_x = x_c - int(size/2);
    ref_y = y_c - int(size/2);

    for ( int i = 0; i < size; ++i )
    {
        for ( int j = 0; j < size; ++j )
        {
            idx = i*size + j;
            y_ret.at<double>(idx, 0) = ref_y + i;
            x_ret.at<double>(idx, 0) = ref_x + j;
        }
    }

    x = x_ret;
    y = y_ret;
}

bool LossFunction::validDescriptorRegion( double x, double y, int W, int H, int border )
{
    if ( x < border || x >= W - border )
    {
        return false;
    }
    else if ( y < border || y >= H - border )
    {
        return false;
    }
    else
    {
        return true;
    }
}

bool LossFunction::validDescriptorRegion( double x, double y, int kpt_size )
{
    int desc_radius = std::max(this->patchSize, int(std::ceil(kpt_size)/2));
    return validDescriptorRegion( x, y, this->W, this->H, desc_radius + this->reg_size );
}

void LossFunction::computeParaboloidNormalForAll( vector<shared_ptr<KeyPoint2>> matched_kpts1, vector<shared_ptr<KeyPoint2>> matched_kpts2, cv::Mat& img )
{
    int W, H, desc_radius;
    double kpt_x, kpt_y, kpt_size;
    shared_ptr<KeyPoint2> kpt1, kpt2;

    W = img.cols;
    H = img.rows;

    for ( int i = 0; i < matched_kpts1.size(); i++ )
    {
        kpt1 = matched_kpts1[i];
        kpt2 = matched_kpts2[i];
        kpt_x = kpt1->getCoordX();
        kpt_y = kpt1->getCoordY();
        kpt_size = kpt1->getSize();
        // Skips keypoints if local region will not produce all valid descriptors.
        desc_radius = std::max(this->patchSize, int(std::ceil(kpt_size)/2));
        if (validDescriptorRegion(kpt_x, kpt_y, W, H, desc_radius + this->reg_size))
        {
            this->collectDescriptorDistance( img, kpt1, kpt2 );
        }
    }
}


void LossFunction::computeDescriptors(const cv::Mat& img, vector<cv::KeyPoint>& kpt, cv::Mat& desc)
{
    this->orb->compute( img, kpt, desc );
}













KeyPointUpdate::KeyPointUpdate(   cv::Mat& img, double* p, cv::Mat K1, cv::Mat K2, 
                                    shared_ptr<LossFunction> loss_func, 
                                    shared_ptr<Parametrization> parametrization)
{
    this->p = p;
    this->img = img;
    this->loss_func = loss_func;
    this->K1 = K1;
    this->K2 = K2;
    this->parametrization = parametrization;
}

void KeyPointUpdate::PrepareForEvaluation(bool evaluate_jacobians, bool new_evaluation_point)
{
    std::cout << "PREPARE FOR EVALUATION" << std::endl;
    if (new_evaluation_point)
    {
        std::cout << "Re-linearizing" << std::endl;
        cv::Mat v_k_opt;
        shared_ptr<KeyPoint2> kpt1, kpt2;
        /*
        for ( int i = 0; i < 6; ++i )
        {
            std::cout << this->p[i] << ", ";
        }
        */

        for (int i = 0; i < this->m_kpts1.size(); ++i)
        {
            kpt1 = m_kpts1[i];
            kpt2 = m_kpts2[i];

            cv::Mat R, t, y_k, x_k, E_matrix, F_matrix, v_k_opt;

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
            y_k = kpt1->getLoc();
            x_k = kpt2->getLoc();


            parametrization->composeRMatrixAndTParam( p_vec, R, t );
            //std::cout << "R_matrix: " << R << std::endl;
            E_matrix = composeEMatrix( R, t );
            //std::cout << "E_matrix: " << E_matrix << std::endl;
            F_matrix = fundamentalFromEssential( E_matrix, this->K1, this->K2 );
            //std::cout << "F_matrix: " << F_matrix << std::endl;
            //std::cout << "A: " << kpt1->getDescriptor("quad_fit") << std::endl;

            // TODO:: REVERT THIS AS WELL
            GJET::epipolarConstrainedOptimization( F_matrix, kpt1->getDescriptor("quad_fit"), x_k, y_k, v_k_opt );
            //GJET::reprojectionError(F_matrix, x_k, y_k, v_k_opt);

            //Updating the keypoint
            kpt1->setDescriptor(v_k_opt, "v_k_opt");
            this->logOptLoc(kpt1);


            // TODO: Remove later when no longer usefull
            //KeyPointUpdate::logKptState( kpt1, F_matrix );


            this->updateKeypoint(kpt1, this->img);

            // Re-linearizing
            //TODO:: REVERT THIS
            loss_func->collectDescriptorDistance( this->img, kpt1, kpt2 );
        }
    }
}

bool KeyPointUpdate::updateKeypoint( std::shared_ptr<KeyPoint2> kpt, const cv::Mat& img )
{   
    // There is no continuity, the original descriptor is being kept to match with next image!!!
    int W, H;
    double x_update, y_update, scale, desc_radius;
    cv::Mat desc, v_k_opt;

    W = img.cols;
    H = img.rows;

    v_k_opt = kpt->getDescriptor("v_k_opt");
    scale = this->calculateScale(v_k_opt);
    x_update = kpt->getCoordX() + scale*v_k_opt.at<double>(0,0);
    y_update = kpt->getCoordY() + scale*v_k_opt.at<double>(1,0);
    //x_update = kpt->getCoordX() + v_k_opt.at<double>(0,0);
    //y_update = kpt->getCoordY() + v_k_opt.at<double>(1,0);

    if ( loss_func->validDescriptorRegion( x_update, y_update,kpt->getSize() ) )
    {
        kpt->setCoordx(x_update);
        kpt->setCoordy(y_update);
        cv::KeyPoint kpt_cv = cv::KeyPoint(x_update, y_update, kpt->getSize());
        vector<cv::KeyPoint> kpt_cv_vec{kpt_cv};

        loss_func->computeDescriptors(img, kpt_cv_vec, desc);

        kpt->setDescriptor(desc.row(0), loss_func->orb_non_rot);
        return true;
    }
    else
    {
        return false;
    }
}

void KeyPointUpdate::moveKptToOptLoc()
{
    cv::Mat y_k_opt;
    shared_ptr<KeyPoint2> kpt1;
    for (int i = 0; i < this->m_kpts1.size(); ++i)
    {
        kpt1 = this->m_kpts1[i];
        y_k_opt = kpt1->getDescriptor("y_k_opt");
        kpt1->setCoordx(y_k_opt.at<double>(0,0));
        kpt1->setCoordy(y_k_opt.at<double>(1,0));
    }
}


void KeyPointUpdate::addEvalKpt(   std::shared_ptr<KeyPoint2> kpt1,
                                    std::shared_ptr<KeyPoint2> kpt2)
{
    this->m_kpts1.push_back(kpt1);
    this->m_kpts2.push_back(kpt2);
}

double KeyPointUpdate::calculateScale(cv::Mat& v_k_opt)
{
    double v_norm = cv::norm(v_k_opt);
    if ( v_norm < this->step_size )
    {
        return 1;
    }
    else
    {
        return (this->step_size)/v_norm;
    }
}

void KeyPointUpdate::logOptLoc( std::shared_ptr<KeyPoint2> kpt )
{
    cv::Mat v_k_opt;
    v_k_opt = kpt->getDescriptor("v_k_opt");

    cv::Mat y_k_opt = (cv::Mat_<double>(3,1) << 
                        kpt->getCoordX() + v_k_opt.at<double>(0,0), 
                        kpt->getCoordY() + v_k_opt.at<double>(1,0), 
                        1);
    kpt->setDescriptor(y_k_opt, "y_k_opt");
}

void KeyPointUpdate::logKptState( std::shared_ptr<KeyPoint2> kpt, cv::Mat F_matrix )
{
    /*
    Arguments:
        kpt:    Keypoint which is wanted to log the current state of.
    Effect:
        Stores the current position of the keypoint, local hamming distances and
        current A matrix in the keypoint 'descriptor' map. And stores them as:
        log_cnt = x
        loc_log_x
        hamming_log_x
        quad_fit_log_x
    */

    int log_nr;
    cv::Mat log_cnt, uv, hamming, A, v_k_opt;


    if ( kpt->isDescriptor("log_cnt") )
    {
        log_cnt = kpt->getDescriptor("log_cnt");
        log_cnt.at<double>(0,0) += 1;
    }
    else
    {
        log_cnt = cv::Mat::zeros(1,1,CV_64F);
    }

    // Get current state
    log_nr = log_cnt.at<double>(0,0);
    uv = kpt->getLoc().clone();
    v_k_opt = kpt->getDescriptor("v_k_opt").clone();
    A = kpt->getDescriptor("quad_fit").clone();
    hamming = kpt->getDescriptor("hamming").clone();

    // Saving logged state
    kpt->setDescriptor(log_cnt, "log_cnt");
    kpt->setDescriptor(uv, "loc_from_log" + std::to_string(log_nr));
    kpt->setDescriptor(v_k_opt, "v_k_opt_log" + std::to_string(log_nr));
    kpt->setDescriptor(A, "quad_fit_log" + std::to_string(log_nr));
    kpt->setDescriptor(hamming, "hamming_log" + std::to_string(log_nr));
    kpt->setDescriptor(F_matrix.clone(), "F_matrix_log" + std::to_string(log_nr));
}