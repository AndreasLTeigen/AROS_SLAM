#include <cmath>
#include <opencv2/opencv.hpp>

#include"descriptorDistribution.hpp"

using std::string;
using std::vector;

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::high_resolution_clock;

using cv::Mat;
using cv::KeyPoint;
using cv::Ptr;

std::vector<cv::KeyPoint> DescDistribExtractor::generateNeighbourhoodKpts( cv::KeyPoint kpt, int reg_size )
{
    /*
    Arguments:
        kpt:        Detected keypoint, center of the local neighbourhood.
        reg_size:   Length of edge in the local neighbourhood (square).
    Returns:
        local_kpts: Keypoints in neighbourhood around <kpt>
    */
    //TODO: Check how the descriptor is computed for keypoints with no orientation in the orb detector. This might cause a problem.
    int idx;
    float ref_x, ref_y, x, y, size;
    vector<cv::KeyPoint> local_kpts(reg_size*reg_size);

    ref_x = kpt.pt.x - reg_size/2; 
    ref_y = kpt.pt.y - reg_size/2;
    
    #pragma omp parallel for
    for ( int row_i = 0; row_i < reg_size; ++row_i )
    {
        y = ref_y + row_i;
        for ( int col_j = 0; col_j < reg_size; ++col_j )
        {
            idx = row_i*reg_size + col_j;
            x = ref_x + col_j;
            size = kpt.size;
            local_kpts[idx] = cv::KeyPoint(x, y, size);
        }
    }

    local_kpts.erase(local_kpts.begin()+int(reg_size/2));       // Removing the central keypoint already calculated
    
    return local_kpts;
}

std::vector<cv::KeyPoint> DescDistribExtractor::generateNeighbourhoodKpts( vector<cv::KeyPoint> kpts, int reg_size )
{
    /*
    Arguments:
        kpt:        List of detected keypoint, centers of the local neighbourhoods.
        reg_size:   Length of edge in the local neighbourhood (square).
    Returns:
        local_kpts: Keypoints in neighbourhoods around all <kpt>s.
    */
    //TODO: Check how the descriptor is computed for keypoints with no orientation in the orb detector. This might cause a problem.
    int idx;
    float ref_x, ref_y, x, y, size;
    //vector<cv::KeyPoint> local_kpts((reg_size*reg_size-1)*kpts.size());
    vector<cv::KeyPoint> local_kpts;
    
    #pragma omp parallel for
    for ( cv::KeyPoint kpt : kpts )
    {
        ref_x = kpt.pt.x - reg_size/2; 
        ref_y = kpt.pt.y - reg_size/2;
        for ( int row_i = 0; row_i < reg_size; ++row_i )
        {
            y = ref_y + row_i;
            for ( int col_j = 0; col_j < reg_size; ++col_j )
            {
                if (row_i != int(reg_size/2) || col_j != int(reg_size/2)) // Let every keypoint but the one we already have be added to the list
                {
                    idx = row_i*reg_size + col_j;
                    x = ref_x + col_j;
                    size = kpt.size;
                    local_kpts.push_back(cv::KeyPoint(x,y,size));
                    //local_kpts[idx] = cv::KeyPoint(x, y, size);
                }
                else
                {
                    local_kpts.push_back(kpt);
                }
            }
        }
    }
    
    return local_kpts;
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

vector<Mat> DescDistribExtractor::sortDescs( vector<cv::KeyPoint>& kpts, vector<cv::KeyPoint>& dummy_kpts, Mat& desc, int reg_size )
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

Mat DescDistribExtractor::computeHammingDistance( Mat& target_desc, Mat& region_descs, int N )
{
    /*
    Arguments:
        target_desc:        Descriptor all other descriptors should be calculated the distance to.
        descs:              All other descriptors.
        N:                  Number of descriptors.
    Returns:
        desc_dists:         Hamming distance between <target_desc> and all descriptors in <descs>
    */

    Mat hamming_dists = Mat::zeros(1, N, CV_64F);
    for ( int i = 0; i < N; ++i )
    {
        hamming_dists.at<double>(0, i) = cv::norm(target_desc, region_descs.row(i));
    }
    return hamming_dists;
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
    Mat desc, center_desc, hamming_dists;

    auto detect_start = high_resolution_clock::now();

    int N_region_descs = this->reg_size*this->reg_size;

    orb->detect( img, kpts );

    //Generate all dummy keypoints
    vector<cv::KeyPoint> dummy_kpts = generateNeighbourhoodKpts(kpts, this->reg_size);
    std::cout << "Dummy points: " << dummy_kpts.size() << std::endl;
    orb->compute( img, dummy_kpts, desc );
    //std::cout << "Dummy points: " << dummy_kpts.size() << std::endl;
    //vector<Mat> region_descs = sortDescs( kpts, dummy_kpts, center_desc, desc, this->reg_size);

    //orb->compute( img, kpts, desc );
    std::cout << "Num descriptors: " << desc.size() << std::endl;
    auto register_start = high_resolution_clock::now();

    frame->registerKeypoints( kpts, center_desc );

    auto full_end = high_resolution_clock::now();


    auto ms1 = duration_cast<milliseconds>(register_start-detect_start);
    auto ms3 = duration_cast<milliseconds>(full_end-register_start);

    std::cout << "Extract: " << ms1.count() << "ms" << std::endl;
    std::cout << "Registration: " << ms3.count() << "ms" << std::endl;
}