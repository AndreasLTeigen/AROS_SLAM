#include <cmath>
#include <memory>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>

#include "../../util/util.hpp"
#include "descriptorDistribution.hpp"

using std::string;
using std::vector;

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::high_resolution_clock;

using cv::Mat;
using cv::KeyPoint;
using cv::Ptr;

bool DescDistribExtractor::validDescriptorRegion( int x, int y, int W, int H, int border )
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

std::vector<cv::KeyPoint> DescDistribExtractor::generateNeighbourhoodKpts( vector<cv::KeyPoint>& kpts, Mat& img )
{
    /*
    Arguments:
        kpt:        List of detected keypoint, centers of the local neighbourhoods.
        reg_size:   Length of edge in the local neighbourhood (square).
    Returns:
        local_kpts: Keypoints in neighbourhoods around all <kpt>s.
    */
    //TODO: Check how the descriptor is computed for keypoints with no orientation in the orb detector. This might cause a problem.

    int idx, W, H, desc_radius;
    float ref_x, ref_y, x, y, size;
    vector<int> removal_kpts;
    vector<cv::KeyPoint> local_kpts;
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
            removal_kpts.push_back(n);
            continue;
        }

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

    for ( int remove_idx : removal_kpts )
    {
        kpts.erase(kpts.begin() + remove_idx);
    }
    
    return local_kpts;
}

void DescDistribExtractor::generateCoordinateVectors(double x_c, double y_c, int size, Mat& x, Mat& y)
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

vector<Mat> DescDistribExtractor::sortDescsN2( vector<cv::KeyPoint>& kpts, vector<cv::KeyPoint>& dummy_kpts, Mat& desc, int reg_size )
{
    // TODO: INCREASE THE SPEED OF THIS ALGORITHM WITH RANGE SEARCH PROBLEM SOLUTION
    int x_c, y_c, x_d, y_d, x_dist, y_dist;
    double r = double(reg_size)/2;
    cv::KeyPoint c_kpt, d_kpt;

    vector<Mat> region_desc(kpts.size());


    for ( int i = 0; i < dummy_kpts.size(); ++i )
    {
        d_kpt = dummy_kpts[i];
        x_d = d_kpt.pt.x;
        x_d = d_kpt.pt.y;

        for ( int j = 0; j < kpts.size(); ++j )
        {
            c_kpt = kpts[j];
            x_c = c_kpt.pt.x;
            y_c = c_kpt.pt.y;

            x_dist = std::abs(x_c - x_d);
            y_dist = std::abs(y_c - y_d);

            if ( x_dist < r && y_dist < r )
            {
                region_desc[j].push_back(desc.row(i));
                break;
            }
        }
    }
    return region_desc;
}

void DescDistribExtractor::sortDescsOrdered(Mat& desc, vector<Mat>& desc_ordered, int reg_size)
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

void DescDistribExtractor::getCenterDesc( vector<Mat>& desc_ordered, Mat& desc_center )
{
    int K = desc_ordered[0].rows;
    for (int i = 0; i < desc_ordered.size(); i++)
    {
        desc_center.push_back(desc_ordered[i].row(int(K/2)));
    }

}

Mat DescDistribExtractor::computeHammingDistance( Mat& target_desc, Mat& region_descs )
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

Mat DescDistribExtractor::computeHammingDistanceAnalysis( cv::KeyPoint target_kpt, Mat& target_desc, vector<cv::KeyPoint> region_kpt, Mat& region_descs )
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
        std::cout << "Kpt sub nr: " << i << std::endl;
        std::cout << "Target kpt: " << target_kpt.pt << std::endl;
        std::cout << "region kpt: " << region_kpt[i].pt << std::endl;
        hamming_dists.at<double>(0, i) = cv::norm(target_desc, region_descs.row(i), cv::NORM_HAMMING);
        std::cout << "target_desc: " << target_desc << std::endl;
        std::cout << "reg_desc: " << region_descs.row(i) << std::endl;
        std::cout << "Hamming dist: " << hamming_dists.at<double>(0, i) << std::endl;
        std::cout << cv::norm(target_desc, region_descs.row(i)) << std::endl;
    }
    std::cout << "##############################################" << std::endl;
    return hamming_dists;
}

std::vector<cv::KeyPoint> DescDistribExtractor::generateDenseKeypoints(cv::Mat& img, float kpt_size)
{
    int border = kpt_size/2 + 1;
    vector<cv::KeyPoint> dense_kpts;

    for (int y = border; y < img.rows-border; y++)
    {
        for (int x = border; x < img.cols-border; x++)
        {
            dense_kpts.push_back(cv::KeyPoint(float(x), float(y), kpt_size));
        }
    }
    return dense_kpts;
}

void DescDistribExtractor::testPrintKeypointOrdering(vector<cv::KeyPoint> dummy_kpts, int kpt_nr)
{
    int N = this->reg_size*this->reg_size;
    cv::KeyPoint kpt;

    std::cout << "Kpt nr: " << kpt_nr << std::endl;
    for ( int i = 0; i < this->reg_size; i++ )
    {
        for ( int j = 0; j < this->reg_size; j++ )
        {
            kpt = dummy_kpts[kpt_nr*N + i*this->reg_size + j];
            std::cout << "|" << kpt.pt.y << ", " << kpt.pt.x << "|";
        }
        std::cout << "\n";
    }
    std::cout << "//////////////////////////////////////////////////////////////" << std::endl;
}

void DescDistribExtractor::printLocalHammingDist( vector<Mat> hamming_dists, int reg_size )
{
    for ( int n = 0; n < hamming_dists.size(); n++ )
    {
        for ( int i = 0; i < reg_size; i++ )
        {
            for ( int j = 0; j < reg_size; j++ )
            {
                std::cout << " | " << hamming_dists[n].col(i*reg_size + j);
            }
            std::cout << " | " << std::endl;
        }
        std::cout << "####################################" << std::endl;
    }
}

Mat DescDistribExtractor::generateKeypointCoverageMap(vector<cv::KeyPoint> kpts, int H, int W)
{
    int x, y;
    Mat c_map = Mat::zeros(H, W, CV_8UC1);

    for ( cv::KeyPoint kpt : kpts )
    {
        x = kpt.pt.x;
        y = kpt.pt.y;
        c_map.at<uchar>(y,x) = c_map.at<uchar>(y,x) + 256/4;
    }
    return c_map;
}

void DescDistribExtractor::registerFrameKeypoints( std::shared_ptr<FrameData> frame, std::vector<cv::KeyPoint>& kpts, cv::Mat& desc, cv::Mat& center_desc, std::vector<cv::Mat>& A, std::vector<cv::Mat>& desc_hamming_dist )
{
    /* Converts and registers vector<cv::KeyPoint> into the AVG Keypoint class
       and saves it in the frameData class. For more details, see design document*/

    Mat hamming_dist;

    #pragma omp parallel for
    for ( int n = 0; n < kpts.size(); ++n )
    {
        std::shared_ptr<KeyPoint2> keypoint = std::make_shared<KeyPoint2>(n, kpts[n], frame->getFrameNr(), desc.row(n));
        Mat center_loc = (cv::Mat_<double>(2,1) << kpts[n].pt.y, kpts[n].pt.x);
        hamming_dist = desc_hamming_dist[n].reshape(this->reg_size, this->reg_size);
        keypoint->setDescriptor(center_loc, "center_loc");
        keypoint->setDescriptor(center_desc, "center_desc");
        keypoint->setDescriptor(hamming_dist, "desc_hamming_dist");
        keypoint->setDescriptor(A[n], "quad_fit");
        frame->addKeypoint(keypoint);
    }
}

void DescDistribExtractor::extract( cv::Mat& img, std::shared_ptr<FrameData> frame, std::shared_ptr<Map3D> map_3d )
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
    Mat desc, rot_desc;

    auto detect_start = high_resolution_clock::now();

    int N = this->reg_size*this->reg_size;

    orb->detect( img, kpts );

    //Generate all dummy keypoints
    //std::cout << "Predicted Dummy points: " << kpts.size() * N << std::endl;
    vector<cv::KeyPoint> dummy_kpts = this->generateNeighbourhoodKpts(kpts, img);

    //std::cout << "Dummy points: " << dummy_kpts.size() << std::endl;
    orb->compute( img, dummy_kpts, desc );
    //std::cout << "Descriptors: " << dummy_kpts.size() << std::endl;

    vector<Mat> desc_ordered;
    this->sortDescsOrdered(desc, desc_ordered, this->reg_size);

    Mat desc_center;
    this->getCenterDesc( desc_ordered, desc_center );

    Mat x, y, z;
    Mat target_desc;
    vector<Mat> hamming_dists(kpts.size());
    vector<Mat> A(kpts.size());                     // Quadratic fittings for each keypoint neighbourhood.
    for ( int i = 0; i < kpts.size(); i++)
    {
        target_desc = desc_center.row(i);
        hamming_dists[i] = computeHammingDistance(target_desc, desc_ordered[i]);
        this->generateCoordinateVectors(kpts[i].pt.x, kpts[i].pt.y, this->reg_size, x, y);
        z = hamming_dists[i].t();
        A[i] = fitQuadraticForm(x, y, z);
    }

    //this->printLocalHammingDist(hamming_dists, this->reg_size);
    
    orb->compute( img, kpts, rot_desc );
    
    std::cout << "Num descriptors: " << desc.rows << std::endl;
    auto register_start = high_resolution_clock::now();

    this->registerFrameKeypoints( frame, kpts, rot_desc, desc_center, A, hamming_dists );
    //frame->registerKeypoints( kpts, desc_center );

    auto full_end = high_resolution_clock::now();


    auto ms1 = duration_cast<milliseconds>(register_start-detect_start);
    auto ms3 = duration_cast<milliseconds>(full_end-register_start);

    std::cout << "Extract: " << ms1.count() << "ms" << std::endl;
    std::cout << "Registration: " << ms3.count() << "ms" << std::endl;
}