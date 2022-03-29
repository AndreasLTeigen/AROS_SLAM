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
    cv::Mat A_k, b_k, c_k, F_d, F_d_x, KKT, q_31, b_k_neg, y1_k, x1_k;
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


// ################ Collecting descriptor differences ####################

void GJET::collectDescriptorDistances( cv::Mat& img, shared_ptr<FrameData> frame1, shared_ptr<FrameData> frame2 )
{
    /*
    Arguments:
        img:    Target image for keypoint detection.
        frame:  <FrameData> to fill with keypoint information.
        map_3d: Inherited variable, not needed for this function.
    
    Effect:
        Computes the best keypoints in the image, computes the descriptors for every pixel
            in an area around that keypoint, computes the hemming distance between the central
            keypoint descriptor and all descriptors in the region, stores this as an extended
            descriptor for the keypoint.
    */
    cv::KeyPoint kpt;
    vector<cv::KeyPoint> kpts;
    vector<shared_ptr<KeyPoint2>> matched_kpts;
    Mat desc, rot_desc;

    auto detect_start = high_resolution_clock::now();

    int N = this->reg_size*this->reg_size;

    // Getting only the matched keypoints and converting them to cv::KeyPoint format.
    matched_kpts = frame1->getMatchedKeypoints( frame2->getFrameNr() );
    kpts = FrameData::compileCVKeypoints( matched_kpts );



    //Generate all dummy keypoints
    vector<cv::KeyPoint> dummy_kpts = this->generateNeighbourhoodKpts(kpts, img);

    orb->compute( img, dummy_kpts, desc );

    vector<Mat> desc_ordered;
    this->sortDescsOrdered(desc, desc_ordered, this->reg_size);

    Mat desc_center;
    this->getCenterDesc( desc_ordered, desc_center );

    Mat x, y, z, test;
    Mat target_desc;
    vector<Mat> hamming_dists(kpts.size());
    vector<Mat> A(kpts.size());                     // Quadratic fittings for each keypoint neighbourhood.
    for ( int i = 0; i < kpts.size(); i++)
    {
        target_desc = desc_center.row(i);
        hamming_dists[i] = this->computeHammingDistance(target_desc, desc_ordered[i]);
        this->generateCoordinateVectors(kpts[i].pt.x, kpts[i].pt.y, this->reg_size, x, y);
        z = hamming_dists[i].t();
        A[i] = fitQuadraticForm(x, y, z);
    }
    
    //orb->compute( img, kpts, rot_desc );
    
    std::cout << "Num descriptors: " << desc.rows << std::endl;
    auto register_start = high_resolution_clock::now();

    //this->registerFrameKeypoints( frame, kpts, rot_desc, desc_center, A, hamming_dists );
    this->registerDDInfo( frame1, frame2, desc_center, A );

    auto full_end = high_resolution_clock::now();


    auto ms1 = duration_cast<milliseconds>(register_start-detect_start);
    auto ms3 = duration_cast<milliseconds>(full_end-register_start);

    std::cout << "Extract: " << ms1.count() << "ms" << std::endl;
    std::cout << "Registration: " << ms3.count() << "ms" << std::endl;
}

std::vector<cv::KeyPoint> GJET::generateNeighbourhoodKpts( vector<cv::KeyPoint>& kpts, Mat& img )
{
    /*
    Arguments:
        kpt:        List of detected keypoint, centers of the local neighbourhoods.
        reg_size:   Length of edge in the local neighbourhood (square).
    Returns:
        local_kpts: Keypoints in neighbourhoods around all <kpt>s.
    */
    //TODO: Check how the descriptor is computed for keypoints with no orientation in the orb detector. This might cause a problem.

    int W, H, desc_radius;
    float ref_x, ref_y, x, y, size;
    vector<int> removal_kpts;
    vector<cv::KeyPoint> center_kpts, local_kpts;
    W = img.cols;
    H = img.rows;
    
    #pragma omp parallel for
    for ( int n = 0; n < kpts.size(); ++n )
    {
        cv::KeyPoint kpt = kpts[n];
        // Skips keypoints if local region will not produce all valid descriptors.
        desc_radius = std::max(this->patchSize, int(std::ceil(kpt.size)/2));
        if ( !validDescriptorRegion(kpt.pt.x, kpt.pt.y, W, H, desc_radius + this->reg_size) )
        {
            continue;
        }
        else
        {
            center_kpts.push_back(kpt);
            ref_x = kpt.pt.x - reg_size/2; 
            ref_y = kpt.pt.y - reg_size/2;
            for ( int row_i = 0; row_i < reg_size; ++row_i )
            {
                y = ref_y + row_i;
                for ( int col_j = 0; col_j < reg_size; ++col_j )
                {
                    x = ref_x + col_j;
                    size = kpt.size;
                    local_kpts.push_back(cv::KeyPoint(x,y,size));
                }
            }
        }
    }

    kpts = center_kpts;
    
    return local_kpts;
}

void GJET::sortDescsOrdered(Mat& desc, vector<Mat>& desc_ordered, int reg_size)
{
    /*
    Arguments:
        desc:       Descriptors for all keypoints in all local regions, stored in reg_size*reg_size chunks.
        reg_size:   Size of the local neighbourhood.
    Returns:
        desc_ordered:   All descriptors belonging to neighbourhood[i] stored as vector element i.
    Assumption:
        <desc> is ordered in chunks of size reg_size*reg_size belonging to each keypoint.
    */
    int K = reg_size*reg_size;

    for ( int n = 0; n < desc.rows/K; n++)
    {
        Mat neighborhood_desc;
        for ( int i = 0; i < reg_size; i++ )
        {
            for ( int j = 0; j < reg_size; j++ )
            {
                neighborhood_desc.push_back(desc.row( n*K + i*reg_size + j ));
            }
        }
        desc_ordered.push_back(neighborhood_desc);
    }
}

void GJET::getCenterDesc( vector<Mat>& desc_ordered, Mat& desc_center )
{
    int K = desc_ordered[0].rows;
    for (int i = 0; i < desc_ordered.size(); i++)
    {
        desc_center.push_back(desc_ordered[i].row(int(K/2)));
    }
}

Mat GJET::computeHammingDistance( Mat& target_desc, Mat& region_descs )
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

void GJET::generateCoordinateVectors(double x_c, double y_c, int size, Mat& x, Mat& y)
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

void GJET::registerDDInfo( shared_ptr<FrameData> frame1, shared_ptr<FrameData> frame2, cv::Mat& center_desc, std::vector<cv::Mat>& A )
{
    vector<shared_ptr<KeyPoint2>> kpts = frame1->getMatchedKeypoints( frame2->getFrameNr() );

    #pragma omp parallel for
    for ( int i = 0; i < kpts.size(); ++i )
    {
        kpts[i]->setDescriptor( A[i], "quad_fit" );
    }
}

bool GJET::validDescriptorRegion( int x, int y, int W, int H, int border )
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