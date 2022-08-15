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
    ECOptSolver(    const cv::Mat K1, 
                    const cv::Mat K2,
                    const shared_ptr<KeyPoint2> kpt1,
                    const shared_ptr<KeyPoint2> kpt2,
                    const shared_ptr<Parametrization> parametrization, 
                    const shared_ptr<LossFunction> loss_func )
                {
                    K1_ = K1;
                    K2_ = K2;
                    kpt1_ = kpt1,
                    kpt2_ = kpt2,
                    loss_func_ = loss_func;
                    parametrization_ = parametrization;


                    cv::Mat residual_ = cv::Mat::zeros(1,1,CV_64F);
                    residual_.at<double>(0,0) = 10000;

                    kpt1_->setDescriptor(residual_, "residual");
                }
    
    bool operator()( const double* p, const double* Y, double* residual ) const
    {
        if ( !loss_func_->validKptLoc( Y[0], Y[1], kpt1_->getSize() ) )
        {
            residual[0] = kpt1_->getDescriptor("residual").at<double>(0,0);
        }
        else
        {  
            cv::Mat A, R, t, y_k, x_k, E_matrix, F_matrix, v_k_opt;

            // TODO: The variables below are calculated for every single keypoint, but this shouldn't have to be the case.
            //          See if there is some place this can be calculated only once per test change of the parameters.
            
            int n = 6;
            vector<double> p_vec(p, p + n);

            y_k = (cv::Mat_<double>(3,1)<<  Y[0],
                                            Y[1],
                                            1);
            x_k = (cv::Mat_<double>(3,1)<<  kpt2_->getCoordX(),
                                            kpt2_->getCoordY(),
                                            1);

            parametrization_->composeRMatrixAndTParam( p_vec, R, t );
            E_matrix = composeEMatrix( R, t );
            F_matrix = fundamentalFromEssential( E_matrix, K1_, K2_ );

            loss_func_->linearizeLossFunctionV2( y_k, kpt2_, A);

            residual[0] = loss_func_->calculateKptLoss( F_matrix, A, y_k, x_k, v_k_opt );


            cv::Mat residual_ = cv::Mat::zeros(1,1,CV_64F);
            residual_.at<double>(0,0) = residual[0];
            kpt1_->setDescriptor(residual_, "residual");
        }

        return true;
    }

    cv::Mat K1_, K2_, A_k_;
    shared_ptr<KeyPoint2> kpt1_, kpt2_;
    shared_ptr<LossFunction> loss_func_;
    shared_ptr<Parametrization> parametrization_;
};

/*
struct GJETSolver
{
    GJETSolver(const cv::Mat K1, const cv::Mat K2, const shared_ptr<KeyPoint2> kpt1, 
                        const shared_ptr<KeyPoint2> kpt2, const shared_ptr<Parametrization> parametrization, const shared_ptr<LossFunction> loss_func)
                    : ecopt_solver(new ceres::NumericDiffCostFunction<ECOptSolver, ceres::CENTRAL, 1, 6>(
                                                new ECOptSolver(K1, K2, kpt1, kpt2, parametrization, loss_func))) {}
    
    template <typename T>
    bool operator()(const T* p, T* residual) const
    {
        return ecopt_solver(p, residual);
    }

    private:
        ceres::CostFunctionToFunctor<1,6> ecopt_solver;
};
*/
















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

    vector<shared_ptr<KeyPoint2>> matched_kpts1 = frame1->getMatchedKeypoints( frame2->getFrameNr() );
    vector<shared_ptr<KeyPoint2>> matched_kpts2 = frame2->getMatchedKeypoints( frame1->getFrameNr() );


    //shared_ptr<LossFunction> loss_func = std::make_shared<LossFunction>(img);
    shared_ptr<LossFunction> loss_func = std::make_shared<DJETLoss>(img, matched_kpts1, matched_kpts2);
    //shared_ptr<LossFunction> loss_func = std::make_shared<ReprojectionLoss>(img);

    // ------ CERES test -------------
    int N = matched_kpts1.size();
    cv::Mat A_d_k, x_k, y_k;
    shared_ptr<KeyPoint2> kpt1, kpt2;

    // Initializing variables
    vector<double> p_init = rel_pose->getParametrization( this->paramId )->getParamVector();
    std::cout << *rel_pose->getParametrization( this->paramId ) << "\n";
    double p[p_init.size()];
    vector<double*> points2D;
    
    //Filling p with values from p_init
    std::copy(p_init.begin(), p_init.end(), p);

    shared_ptr<StdParam> parametrization = std::make_shared<StdParam>();

    KeyPointUpdate itUpdate( img, p, frame1->getKMatrix(), frame2->getKMatrix(), loss_func, parametrization );  
    ceres::Problem::Options problem_options;
    //problem_options.evaluation_callback = &itUpdate; 
    ceres::Problem problem(problem_options);
    
    for ( int n = 0; n < N; ++n )
    {
        kpt1 = matched_kpts1[n];
        kpt2 = matched_kpts2[n];

        if( loss_func->validKptLoc( kpt1->getCoordX(), kpt1->getCoordY(), kpt1->getSize()) )
        {
            double point2D[2] = {kpt1->getCoordX(), kpt1->getCoordY()};
            points2D.push_back(point2D);

            ceres::CostFunction* cost_function = new ceres::NumericDiffCostFunction<ECOptSolver, ceres::CENTRAL, 1, 6, 2>(
                new ECOptSolver(frame1->getKMatrix(), frame2->getKMatrix(), kpt1, kpt2, parametrization, loss_func));
            problem.AddResidualBlock(cost_function, nullptr, p, point2D);
        }
        else
        {
            KeyPointUpdate::invalidateMatch( kpt1, kpt2 );
        }
    }

    // Configure the solver
    ceres::Solver::Options options;
    options.use_nonmonotonic_steps = true;
    options.preconditioner_type = ceres::SCHUR_JACOBI;
    //options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 100;
    options.num_threads = 1;
    //options.initial_trust_region_radius = 1;
    //options.max_trust_region_radius = 100;
    //options.logging_type = ceres::LoggingType::SILENT;

    // Solve!
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

    int n = sizeof(p) / sizeof(p[0]);
    vector<double> p_vec(p, p + n);

    rel_pose->setPose( p_vec, this->paramId );
    //rel_pose->setPose( p_init, this->paramId );

    //std::cout << rel_pose->getTMatrix() << std::endl;
    cv::Mat R_matrix = rel_pose->getRMatrix();
    cv::Mat t_vector = rel_pose->gettvector();
    t_vector = normalizeMat(t_vector);
    rel_pose->updatePoseVariables( R_matrix, t_vector );
    rel_pose->updateParametrization(); // This function call can be removed
    std::cout << "p_corr: " << *rel_pose->getParametrization() << "\n";

    F_matrix = fundamentalFromEssential( rel_pose->getEMatrix(), frame1->getKMatrix(), frame2->getKMatrix() );
    //--------------------------------------

    // Last changes to keypoint locations
    //itUpdate.moveKptsToOptLoc(F_matrix, img);
    //FrameData::removeMatchesWithLowConfidence( -0.9, frame1, frame2 );

    itUpdate.updateKeypoints(matched_kpts1, matched_kpts2, points2D);

    matched_kpts1 = frame1->getMatchedKeypoints( frame2->getFrameNr() );
    matched_kpts2 = frame2->getMatchedKeypoints( frame1->getFrameNr() );
    //std::cout << "Total Loss: " << loss_func->calculateTotalLoss(F_matrix, matched_kpts1, matched_kpts2) << "\n" << std::endl;

    // Evaluation
    int old_mean = this->avg_match_score;
    this->n += 1;
    this->n_matches += matched_kpts1.size();
    this->avg_match_score = iterativeAverage(this->avg_match_score, GJET::calculateAverageMatchScore( matched_kpts1, matched_kpts2 ), this->n);
    this->varianceN_match_score = GJET::iterateMatchScoreVarianceN(matched_kpts1, matched_kpts2, this->varianceN_match_score, old_mean, this->avg_match_score);
    this->avg_calculated_descs = iterativeAverage(this->avg_calculated_descs, loss_func->calculated_descs/matched_kpts1.size(), this->n);

    std::cout << "Avg match score: " << avg_match_score << std::endl;
    std::cout << "STD match score: " << std::sqrt(this->varianceN_match_score/this->n_matches) << std::endl;
    std::cout << "Avg calculated_descs: " << this->avg_calculated_descs << std::endl;

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
    F_d = I_s * F_matrix.t();
    F_d_x = F_d*x1_k;
    q_31 = -(y1_k.t()) * F_matrix.t() * (x1_k);
    
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

    /*
    std::cout << "x1_k:" << x1_k << std::endl;
    std::cout << "y1_k:" << y1_k << std::endl;
    std::cout << "F_d:" << F_d << std::endl;
    std::cout << "A_k:" << A_k << std::endl;
    std::cout << "F_d_x:" << F_d_x << std::endl;
    std::cout << "b_k_neg:" << b_k_neg << std::endl;
    std::cout << "q_31:" << q_31 << std::endl;
    std::cout << "v_k_opt:" << v_k_opt << std::endl;
    std::cout << "---------------" << std::endl;
    */


    return GJET::solveQuadraticFormForV( A_k, b_k, c_k, v_k_opt );
}

void GJET::analysis( cv::Mat &img_disp, std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2 )
{
    int random_idx, canvas_h, canvas_w;
    double hamming_dist;
    cv::Mat img1, img2, F_matrix, A, uv, hamming;
    cv::Mat canvas(840, 1400, img_disp.type(), cv::Scalar::all(0));
    shared_ptr<KeyPoint2> kpt1, kpt2;
    vector<shared_ptr<KeyPoint2>> matched_kpts1, matched_kpts2;

    matched_kpts1 = frame1->getMatchedKeypoints( frame2->getFrameNr() );
    matched_kpts2 = frame2->getMatchedKeypoints( frame1->getFrameNr() );

    img1 = frame1->getImg();
    img2 = frame2->getImg();
    cvtColor(img1, img1, cv::COLOR_GRAY2BGR );
    cvtColor(img2, img2, cv::COLOR_GRAY2BGR );

    int border = 30;
    //cv::Size size(31,31);
    cv::Size size(101,101);

    int num_it = int(matched_kpts1[0]->getDescriptor("log_cnt").at<double>(0,0));

    for (int it = 0; it < num_it+1; ++it )
    {

        copyMakeBorder(img_disp, canvas, 0, canvas.rows-img_disp.rows, 0, canvas.cols-img_disp.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0) );
        
        F_matrix = matched_kpts1[0]->getDescriptor("F_matrix_log" + std::to_string(it));
        //std::cout << "F_matrix: " << F_matrix << std::endl;

        srand (time(NULL));
        for ( int i = 0; i < 10; ++i )
        {
            random_idx = rand() % matched_kpts1.size();
            random_idx = i;
            kpt1 = matched_kpts1[random_idx];
            kpt2 = matched_kpts2[random_idx];
            hamming = kpt1->getDescriptor("hamming_log" + std::to_string(it));
            A = kpt1->getDescriptor("quad_fit_log" + std::to_string(it));
            cv::Mat z;
            if (!A.empty())
            {
                uv = kpt1->getDescriptor("loc_from_log" + std::to_string(it));
                z = sampleQuadraticForm(A, cv::Point(uv.at<double>(0,0),uv.at<double>(1,0)), cv::Size(31,31) );
            }
            KeyPoint2::drawEnchancedKeyPoint( canvas, img2, kpt2, cv::Point((border + size.width)*i, 400), size, cv::Mat());
            KeyPoint2::drawKptHeatMapAnalysis( canvas, img1, kpt1, cv::Point((border + size.width)*i, 510), size, F_matrix, kpt2, hamming, it, false, false );
            KeyPoint2::drawKptHeatMapAnalysis( canvas, img1, kpt1, cv::Point((border + size.width)*i, 620), size, F_matrix, kpt2, hamming, it, true, true );
            KeyPoint2::drawKptHeatMapAnalysis( canvas, img1, kpt1, cv::Point((border + size.width)*i, 730), size, F_matrix, kpt2, z, it, true, true );
        }

        cv::imshow("KeyPoint Log iteration: " + std::to_string(it), canvas);
        cv::waitKey(0);
    }
}

double GJET::calculateAverageMatchScore(std::vector<std::shared_ptr<KeyPoint2>> matched_kpts1, 
                                        std::vector<std::shared_ptr<KeyPoint2>> matched_kpts2)
{
    double accum;
    shared_ptr<KeyPoint2> kpt1, kpt2;
    cv::Mat desc1, desc2, hamming;

    for (int i = 0; i < matched_kpts1.size(); ++i)
    {
        kpt1 = matched_kpts1[i];
        kpt2 = matched_kpts2[i];

        desc1 = kpt1->getDescriptor("orb");
        desc2 = kpt2->getDescriptor("orb");

        hamming = computeHammingDistance(desc1, desc2);
        accum += hamming.at<double>(0,0);
    }
    accum = accum / matched_kpts1.size();
    return accum;
}


double GJET::iterateMatchScoreVarianceN(std::vector<std::shared_ptr<KeyPoint2>> matched_kpts1, 
                                        std::vector<std::shared_ptr<KeyPoint2>> matched_kpts2, 
                                        double old_varianceN, double old_mean, double mean)
{
    double accum, x_n;
    shared_ptr<KeyPoint2> kpt1, kpt2;
    cv::Mat desc1, desc2, hamming;

    accum = old_varianceN;
    for (int i = 0; i < matched_kpts1.size(); ++i)
    {
        kpt1 = matched_kpts1[i];
        kpt2 = matched_kpts2[i];

        desc1 = kpt1->getDescriptor("orb");
        desc2 = kpt2->getDescriptor("orb");

        hamming = computeHammingDistance(desc1, desc2);
        x_n = hamming.at<double>(0,0);
        accum = accum + (x_n - old_mean)*(x_n - mean);
    }
    return accum;
}




















// ################ Collecting descriptor differences ####################

LossFunction::LossFunction(cv::Mat& img)
{
    this->W = img.cols;
    this->H = img.rows;
}

LossFunction::LossFunction(int W, int H)
{
    this->W = W;
    this->H = H;
}

int LossFunction::getImgWidth()
{
    return this->W;
}

int LossFunction::getImgHeight()
{
    return this->H;
}

int LossFunction::getPatchSize()
{
    return this->patchSize;
}

void LossFunction::computeDescriptors(const cv::Mat& img, vector<cv::KeyPoint>& kpt, cv::Mat& desc)
{
    this->orb->compute( img, kpt, desc );
}

double LossFunction::calculateTotalLoss(cv::Mat& F_matrix,
                                vector<shared_ptr<KeyPoint2>> matched_kpts1, 
                                vector<shared_ptr<KeyPoint2>> matched_kpts2)
{
    double tot_loss, loss_n;
    cv::Mat v_k_opt;
    shared_ptr<KeyPoint2> kpt1, kpt2;
    for(int i = 0; i < matched_kpts1.size(); ++i)
    {
        kpt1 = matched_kpts1[i];
        kpt2 = matched_kpts2[i];
        loss_n = this->calculateKptLoss(F_matrix, kpt1, kpt2, v_k_opt);
        tot_loss += loss_n*loss_n;
    }
    return tot_loss;
}

bool LossFunction::validDescriptorRegion( double x, double y, int border )
{
    if ( x < border || x >= this->W - border )
    {
        return false;
    }
    else if ( y < border || y >= this->H - border )
    {
        return false;
    }
    else
    {
        return true;
    }
}

int LossFunction::calculateDescriptorRadius(int patch_size, int kpt_size)
{
    return std::max(patch_size, int(std::ceil(kpt_size)/2));
}






DJETLoss::DJETLoss(cv::Mat& img, std::vector<std::shared_ptr<KeyPoint2>>& matched_kpts1, std::vector<std::shared_ptr<KeyPoint2>>& matched_kpts2)
    :LossFunction(img.cols, img.rows)
{
    this->img = img;
    std::vector<std::vector<cv::Mat>> descriptors(this->H, std::vector<cv::Mat>(this->W, cv::Mat()));
    this->descriptor_map = descriptors;
    if (this->precompDescriptors)
    {
        auto t1 = high_resolution_clock::now(); 
        this->precomputeDescriptors( img );
        auto t2 = high_resolution_clock::now();
        auto ms1 = duration_cast<milliseconds>(t2-t1);
        std::cout << "Dense descriptor calculation time: " << ms1.count() << "ms" << std::endl;
    }
}

double DJETLoss::calculateKptLoss(const cv::Mat& F_matrix, const cv::Mat& A_d_k, const cv::Mat& x_k, const cv::Mat& y_k, cv::Mat& v_k_opt)
{
    return GJET::epipolarConstrainedOptimization( F_matrix, A_d_k, x_k, y_k, v_k_opt );
}

double DJETLoss::calculateKptLoss(const cv::Mat& F_matrix, const std::shared_ptr<KeyPoint2> kpt1, const std::shared_ptr<KeyPoint2> kpt2, cv::Mat& v_k_opt)
{
    cv::Mat x_k, y_k, A_d_k;
    y_k = kpt1->getLoc();
    x_k = kpt2->getLoc();
    A_d_k = kpt1->getDescriptor("quad_fit");
    return GJET::epipolarConstrainedOptimization( F_matrix, A_d_k, x_k, y_k, v_k_opt );
}

bool DJETLoss::validKptLoc( double x, double y, int kpt_size )
{
    int desc_radius = this->calculateDescriptorRadius(this->patchSize, kpt_size);
    return this->validDescriptorRegion( x, y, desc_radius + this->reg_size );
}

bool DJETLoss::updateKeypoint( std::shared_ptr<KeyPoint2> kpt, const cv::Mat& img, double x_update, double y_update )
{
    // If the update will lead to a keypoint that is in bounds, the keypoint is updated and the function returns true.
    // If the update causes the keypoint to be out of bounds, the function returns false.

    if ( this->validKptLoc( x_update, y_update, kpt->getSize() ) )
    {
        kpt->setCoordx(x_update);
        kpt->setCoordy(y_update);
        return true;
    }
    else
    {
        return false;
    }
}

void DJETLoss::linearizeLossFunctionV2( cv::Mat& y_k, std::shared_ptr<KeyPoint2> kpt2, cv::Mat& A )
{
    this->collectDescriptorDistanceV2( y_k, kpt2, A);
}

void DJETLoss::linearizeLossFunction( cv::Mat& img, std::shared_ptr<KeyPoint2> kpt1, std::shared_ptr<KeyPoint2> kpt2 )
{
    this->collectDescriptorDistance( img, kpt1, kpt2 );
}


void DJETLoss::computeDescriptors(const cv::Mat& img, std::vector<cv::KeyPoint>& kpts, cv::Mat& descs)
{
    // This function can be optimized with a descriptor computation function with no overhead.
    // Assumption: All descriptors are in a valid descriptor range.

    cv::KeyPoint kpt;
    vector<cv::KeyPoint> cmpt_kpts;
    cv::Mat desc, cmpt_descs;

    // Identifying requested descriptors missing from <descriptor_map>.
    for ( int i = 0; i < kpts.size(); ++i )
    {
        kpt = kpts[i];
        desc = this->descriptor_map[kpt.pt.y][kpt.pt.x];
        if (desc.empty())
        {
            this->calculated_descs += 1;
            cmpt_kpts.push_back(kpt);
        }
    }
    
    // Computing missing descriptors.
    LossFunction::computeDescriptors(img, cmpt_kpts, cmpt_descs);
    
    // Filing missing descriptors in <descriptor_map>
    for ( int i = 0; i < cmpt_kpts.size(); ++i )
    {
        kpt = cmpt_kpts[i];
        this->descriptor_map[kpt.pt.y][kpt.pt.x] = cmpt_descs.row(i);
    }

    // Retrieving all requested descriptors
    for ( int i = 0; i < kpts.size(); ++i )
    {
        kpt = kpts[i];
        desc = this->descriptor_map[kpt.pt.y][kpt.pt.x];
        descs.push_back(desc);
    }
}


void DJETLoss::collectDescriptorDistanceV2( cv::Mat& y_k, std::shared_ptr<KeyPoint2> kpt2,cv::Mat& A )
{
    double y_k_x, y_k_y;
    cv::Mat local_descs, target_desc, hamming_dists, x, y, z;
    vector<cv::KeyPoint> local_kpts;

    y_k_x = y_k.at<double>(0,0);
    y_k_y = y_k.at<double>(1,0);

    local_kpts = this->generateLocalKptsV2( y_k_x, y_k_y, kpt2->getSize(), this->img );
    this->computeDescriptors(this->img, local_kpts, local_descs);

    //target_desc = kpt1->getDescriptor(descriptor_name);
    target_desc = kpt2->getDescriptor(this->descriptor_name);
    hamming_dists = computeHammingDistance(target_desc, local_descs);
    this->generateCoordinateVectors(y_k_x, y_k_y, this->reg_size, x, y);
    z = hamming_dists.t();

    A = fitQuadraticForm(x, y, z);
}

void DJETLoss::collectDescriptorDistance( const cv::Mat& img, shared_ptr<KeyPoint2> kpt1, shared_ptr<KeyPoint2> kpt2 )
{
    cv::Mat local_descs, target_desc, A, hamming_dists, x, y, z;
    vector<cv::KeyPoint> local_kpts;

    local_kpts = this->generateLocalKpts( kpt1, img );
    this->computeDescriptors(img, local_kpts, local_descs);

    //target_desc = kpt1->getDescriptor(descriptor_name);
    target_desc = kpt2->getDescriptor(this->descriptor_name);
    hamming_dists = computeHammingDistance(target_desc, local_descs);
    this->generateCoordinateVectors(kpt1->getCoordX(), kpt1->getCoordY(), this->reg_size, x, y);
    z = hamming_dists.t();

    A = fitQuadraticForm(x, y, z);
    kpt1->setDescriptor( A, "quad_fit" );
    kpt1->setDescriptor( z.reshape(1, this->reg_size), "hamming");
}

vector<cv::KeyPoint> DJETLoss::generateLocalKptsV2( double kpt_x, double kpt_y, double kpt_size, const cv::Mat& img )
{
    // Assumes the local region around the keypoint will produce all valid descriptors.
    
    double x, y, ref_x, ref_y;
    vector<cv::KeyPoint> local_kpts;

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
    
    return local_kpts;
}

vector<cv::KeyPoint> DJETLoss::generateLocalKpts( shared_ptr<KeyPoint2> kpt, const cv::Mat& img )
{
    // Assumes the local region around the keypoint will produce all valid descriptors.

    int W, H, desc_radius;
    double ref_x, ref_y, x, y, kpt_x, kpt_y, kpt_size;
    vector<int> removal_kpts;
    vector<cv::KeyPoint> local_kpts;
    W = img.cols;
    H = img.rows;

    kpt_x = kpt->getCoordX();
    kpt_y = kpt->getCoordY();
    kpt_size = kpt->getSize();

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
    
    return local_kpts;
}

//Mat DJETLoss::computeHammingDistance( Mat& target_desc, Mat& region_descs )
//{
    /*
    Arguments:
        target_desc:        Descriptor all other descriptors should be calculated the distance to.
        descs:              All other descriptors.
        N:                  Number of descriptors.
    Returns:
        desc_dists:         Hamming distance between <target_desc> and all descriptors in <descs>
    */
/*
    int N = region_descs.rows;
    Mat hamming_dists = Mat::zeros(1, N, CV_64F);
    for ( int i = 0; i < N; ++i )
    {
        hamming_dists.at<double>(0, i) = cv::norm(target_desc, region_descs.row(i), cv::NORM_HAMMING);
    }
    return hamming_dists;
}
*/

void DJETLoss::generateCoordinateVectors(double x_c, double y_c, int size, Mat& x, Mat& y)
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

void DJETLoss::computeParaboloidNormalForAll( vector<shared_ptr<KeyPoint2>> matched_kpts1, vector<shared_ptr<KeyPoint2>> matched_kpts2, cv::Mat& img )
{
    // This function is no longer called
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
        if (validKptLoc( kpt_x, kpt_y, kpt_size ))
        {
            this->collectDescriptorDistance( img, kpt1, kpt2 );
        }
    }
}


void DJETLoss::precomputeDescriptors( const cv::Mat& img )
{
    // Generating dummy keypoints
    std::cout << "Precomputing dense descriptors..." << std::endl;
    vector<cv::KeyPoint> dense_kpt_list;
    for ( int row_i = 0; row_i < this->H; ++row_i )
    {
        for ( int col_j = 0; col_j < this->W; ++col_j )
        {
            cv::KeyPoint kpt(col_j, row_i, 31);
            dense_kpt_list.push_back(kpt);
        }
    }

    // Computing all descriptors
    cv::Mat desc;
    LossFunction::computeDescriptors(img, dense_kpt_list, desc);

    // Sorting keypoints into array
    cv::KeyPoint kpt;

    for ( int i = 0; i < dense_kpt_list.size(); ++i )
    {
        kpt = dense_kpt_list[i];
        this->descriptor_map[kpt.pt.y][kpt.pt.x] = desc.row(i);
    }
}


void DJETLoss::printKptLoc( vector<cv::KeyPoint> kpts, int rows, int cols )
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

void DJETLoss::printLocalHammingDists( cv::Mat& hamming_dist_arr, int s )
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

void DJETLoss::printDescriptorMapFill()
{
    cv::Mat fill_img = cv::Mat::zeros(this->H, this->W, CV_8UC1);
    for (int row_i = 0; row_i < this->H; ++row_i)
    {
        for (int col_j = 0; col_j < this->W; ++col_j)
        {
            if (!this->descriptor_map[row_i][col_j].empty())
            {
                fill_img.at<uchar>(row_i, col_j) = 255;
            }
        }
    }
    cv::imshow("Descriptor Fill Img", fill_img);
    cv::waitKey(0);
}

void DJETLoss::printCalculatedDescsLog()
{
    std::cout << "Total number of descriptors calculated: " << this->calculated_descs << std::endl;
}



ReprojectionLoss::ReprojectionLoss(cv::Mat& img)
    :LossFunction(img)
{
    
}

double ReprojectionLoss::calculateKptLoss(const cv::Mat& F_matrix, const cv::Mat& A_d_k, const cv::Mat& x_k, const cv::Mat& y_k, cv::Mat& v_k_opt)
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

    return loss;
}

double ReprojectionLoss::calculateKptLoss(const cv::Mat& F_matrix, const std::shared_ptr<KeyPoint2> kpt1, const std::shared_ptr<KeyPoint2> kpt2, cv::Mat& v_k_opt)
{
    double a, b, c, x0, y0, epi_x, epi_y, loss;
    cv::Mat x_k, y_k, epiline;
    y_k = kpt1->getLoc();
    x_k = kpt2->getLoc();

    epiline = F_matrix.t() * x_k;
    //epiline = x_k.t() * F_matrix;
    //std::cout << epiline << std::endl;
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

    return loss;
}

bool ReprojectionLoss::validKptLoc( double x, double y, int kpt_size )
{
    // Keypoint has to be inside a border equal to the descriptor radius.
    int desc_radius = this->calculateDescriptorRadius(this->patchSize, kpt_size);
    return validDescriptorRegion( x, y, desc_radius );
}

bool ReprojectionLoss::updateKeypoint( std::shared_ptr<KeyPoint2> kpt, const cv::Mat& img, double x_update, double y_update )
{
    if ( this->validKptLoc( x_update, y_update, kpt->getSize() ) )
    {
        kpt->setCoordx(x_update);
        kpt->setCoordy(y_update);
        return true;
    }
    else
    {
        return false;
    }
}

void ReprojectionLoss::linearizeLossFunction( cv::Mat& img, std::shared_ptr<KeyPoint2> kpt1, std::shared_ptr<KeyPoint2> kpt2 )
{

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

void KeyPointUpdate::updateKeypoints(   vector<shared_ptr<KeyPoint2>> matched_kpts1, 
                                        vector<shared_ptr<KeyPoint2>> matched_kpts2, 
                                        vector<double*> points2D)
{
    double* point2D;
    shared_ptr<KeyPoint2> kpt1, kpt2;

    int n = 0;
    for ( int i = 0; i < matched_kpts1.size(); ++i )
    {
        kpt1 = matched_kpts1[i];
        kpt2 = matched_kpts2[i];

        if (KeyPointUpdate::validMatch(kpt1, kpt2))
        {
            point2D = points2D[n];
            if (this->loss_func->validKptLoc( point2D[0], point2D[1], kpt1->getSize() ))
            {
                kpt1->setCoordx(point2D[0]);
                kpt1->setCoordy(point2D[1]);
            }
            else
            {
                KeyPointUpdate::invalidateMatch(kpt1, kpt2);
            }
            n += 1;
        }
    }
    std::cout << "111" << std::endl;
    KeyPointUpdate::removeInvalidMatches(matched_kpts1, matched_kpts2);
    std::cout << "222" << std::endl;
}

void KeyPointUpdate::PrepareForEvaluation(bool evaluate_jacobians, bool new_evaluation_point)
{
    /*
    if (new_evaluation_point)
    {
        //std::cout << "Re-linearizing" << std::endl;
        bool in_bounds;
        double tot_loss, loss_n;
        cv::Mat R, t, y_k, x_k, E_matrix, F_matrix, v_k_opt;
        shared_ptr<KeyPoint2> kpt1, kpt2;

        
        
        vector<double> p_vec;
        for ( int i = 0; i < 6; ++i )
        {
            p_vec.push_back(p[i]);
        }
        

        parametrization->composeRMatrixAndTParam( p_vec, R, t );
        E_matrix = composeEMatrix( R, t );
        F_matrix = fundamentalFromEssential( E_matrix, this->K1, this->K2 );


        // Calculate update function
        for (int i = 0; i < this->m_kpts1.size(); ++i)
        {

            kpt1 = m_kpts1[i];
            kpt2 = m_kpts2[i];

            if ( !validMatch(kpt1, kpt2) )
            {
                continue;
            }

            loss_func->linearizeLossFunction( this->img, kpt1, kpt2 );

            loss_n = loss_func->calculateKptLoss( F_matrix, kpt1, kpt2, v_k_opt );
            kpt1->setDescriptor(v_k_opt, "v_k_opt_candidate");


            tot_loss += loss_n*loss_n;
        }
        //std::cout << "Total Loss: " << tot_loss << std::endl;

        // Keypoint positions are only updated if loss is reduced.
        if ( tot_loss < this->best_loss || this->best_loss == -1 )
        {
            this->best_loss = tot_loss;
            for (int i = 0; i < this->m_kpts1.size(); ++i)
            {
                kpt1 = m_kpts1[i];
                kpt2 = m_kpts2[i];

                //Updating the keypoint
                v_k_opt = kpt1->getDescriptor("v_k_opt_candidate");
                kpt1->setDescriptor(v_k_opt, "v_k_opt");
                this->logOptLoc(kpt1);

                // TODO: Remove later when no longer usefull
                KeyPointUpdate::logKptState( kpt1, F_matrix );

                in_bounds = true;//this->updateKeypoint(kpt1, this->img);

                if ( !in_bounds )
                {
                    invalidateMatch( kpt1, kpt2 );
                }
            }
        }
    }
    */
}

bool KeyPointUpdate::updateKeypoint( std::shared_ptr<KeyPoint2> kpt, const cv::Mat& img )
{   
    // There is no continuity, the original descriptor is being kept to match with next image!!!
    bool in_bounds;
    double x_update, y_update, scale, desc_radius;
    cv::Mat v_k_opt;

    v_k_opt = kpt->getDescriptor("v_k_opt");
    scale = this->calculateScale(v_k_opt);
    x_update = kpt->getCoordX() + scale*v_k_opt.at<double>(0,0);
    y_update = kpt->getCoordY() + scale*v_k_opt.at<double>(1,0);

    in_bounds = this->loss_func->updateKeypoint( kpt, img, x_update, y_update );

    return in_bounds;
}

void KeyPointUpdate::moveKptsToOptLoc(cv::Mat& F_matrix, const cv::Mat& img)
{
    cv::Mat y_k_opt;
    bool in_bounds;
    int desc_radius;
    double x_update, y_update;
    shared_ptr<KeyPoint2> kpt1, kpt2;
    for (int i = 0; i < this->m_kpts1.size(); ++i)
    {
        kpt1 = this->m_kpts1[i];
        kpt2 = this->m_kpts2[i];

        if ( !validMatch(kpt1, kpt2) )
        {
            continue;
        }

        y_k_opt = kpt1->getDescriptor("y_k_opt");
        x_update = y_k_opt.at<double>(0,0);
        y_update = y_k_opt.at<double>(1,0);


        desc_radius = LossFunction::calculateDescriptorRadius( loss_func->getPatchSize(), kpt1->getSize() );
        if(this->loss_func->validDescriptorRegion(x_update, y_update, desc_radius))
            //&& cv::norm(kpt1->getDescriptor("v_k_opt")) < this->outlier_threshold)
        {
            // Updating keypoint placement.
            kpt1->setCoordx(x_update);
            kpt1->setCoordy(y_update);
        }
        else
        {
            invalidateMatch(kpt1, kpt2);
        }

        // Updating descriptor.
        cv::Mat desc;
        cv::KeyPoint kpt_cv = cv::KeyPoint(kpt1->getCoordX(), kpt1->getCoordY(), kpt1->getSize());
        vector<cv::KeyPoint> kpt_cv_vec{kpt_cv};
        this->loss_func->computeDescriptors(img, kpt_cv_vec, desc);
        kpt1->setDescriptor(desc.row(0), this->loss_func->descriptor_name);
    }
}

void KeyPointUpdate::moveKptsToOptLocAlt(cv::Mat& F_matrix, const cv::Mat& img)
{
    cv::Mat v_k_opt;
    bool in_bounds;
    int desc_radius;
    double x_update, y_update;
    shared_ptr<KeyPoint2> kpt1, kpt2;
    for (int i = 0; i < this->m_kpts1.size(); ++i)
    {
        kpt1 = this->m_kpts1[i];
        kpt2 = this->m_kpts2[i];

        if ( !validMatch(kpt1, kpt2) )
        {
            continue;
        }


        loss_func->linearizeLossFunction( this->img, kpt1, kpt2 );
        loss_func->calculateKptLoss( F_matrix, kpt1, kpt2, v_k_opt );
        kpt1->setDescriptor(v_k_opt, "v_k_opt");

        in_bounds = this->updateKeypoint(kpt1, this->img);

        if ( !in_bounds )
        {
            invalidateMatch( kpt1, kpt2 );
        }

        // Updating descriptor.
        cv::Mat desc;
        cv::KeyPoint kpt_cv = cv::KeyPoint(kpt1->getCoordX(), kpt1->getCoordY(), kpt1->getSize());
        vector<cv::KeyPoint> kpt_cv_vec{kpt_cv};
        this->loss_func->computeDescriptors(img, kpt_cv_vec, desc);
        kpt1->setDescriptor(desc.row(0), this->loss_func->descriptor_name);
    }
}

void KeyPointUpdate::addEvalKpt( std::shared_ptr<KeyPoint2> kpt1,
                                 std::shared_ptr<KeyPoint2> kpt2)
{
    this->m_kpts1.push_back(kpt1);
    this->m_kpts2.push_back(kpt2);
}

double KeyPointUpdate::calculateScale(cv::Mat& v_k_opt)
{   /*
    double v_norm = cv::norm(v_k_opt);
    if ( v_norm < this->step_size )
    {
        return 1;
    }
    else
    {
        return (this->step_size)/v_norm;
    }
    */
    return 0.1;
}



void KeyPointUpdate::invalidateMatch(std::shared_ptr<KeyPoint2> kpt1, std::shared_ptr<KeyPoint2> kpt2)
{
    kpt1->getHighestConfidenceMatch( kpt2->getObservationFrameNr() )->setValidFlag(false);
}

bool KeyPointUpdate::validMatch(std::shared_ptr<KeyPoint2> kpt1, std::shared_ptr<KeyPoint2> kpt2)
{
    bool valid = kpt1->getHighestConfidenceMatch( kpt2->getObservationFrameNr() )->isValid();
    if ( valid == true )
    {
        return true;
    }
    else
    {
        return false;
    }
}

void KeyPointUpdate::removeInvalidMatches(vector<shared_ptr<KeyPoint2>> matched_kpts1,
                                            vector<shared_ptr<KeyPoint2>> matched_kpts2)
{
    shared_ptr<KeyPoint2> kpt1, kpt2;
    for (int n = 0; n < matched_kpts1.size(); ++n)
    {
        kpt1 = matched_kpts1[n];
        kpt2 = matched_kpts2[n];
        if (!KeyPointUpdate::validMatch(kpt1, kpt2))
        {
            kpt1->removeAllMatches(kpt2->getObservationFrameNr());
        }
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