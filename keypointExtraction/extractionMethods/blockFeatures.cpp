#include <opencv2/opencv.hpp>

#include "blockFeatures.hpp"
#include "../../dataStructures/keypoint.hpp"
#include "../../util/util.hpp"

using std::vector;
using std::shared_ptr;

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::high_resolution_clock;



void BlockFeatures::extract( cv::Mat& img, std::shared_ptr<FrameData> frame, std::shared_ptr<Map3D> map_3d )
{
    /*
    Arguments:
        img:        Image in which to extract features.
        frame:      Frame data in which the <img> belongs to.
    */
    double x, y;
    cv::Mat center;
    shared_ptr<KeyPoint2> kpt;
    vector<shared_ptr<KeyPoint2>> kpts;

    auto detect_start = high_resolution_clock::now();

    int grid_height = this->grid_size.height;
    int grid_width = this->grid_size.width;
    
    double cell_height = img.rows / grid_height;
    double cell_width = img.cols / grid_width;

    for ( int i = 0; i < grid_height; ++i )
    {
        for ( int j = 0; j < grid_width; ++j )
        {
            // Making a keypoint.
            y = cell_height * (i + 0.5);
            x = cell_width * (j + 0.5);
            kpt = std::make_shared<KeyPoint2>( i*grid_width + j, x, y, frame->getFrameNr());
            center = (cv::Mat_<double>(2,1) << x, y );
            kpt->setDescriptor(center, "center");

            // Using a part of the image as the kpt descriptor.
            cv::Rect r(x - cell_width*0.5, y - cell_height*0.5, cell_width, cell_height);
            cv::Mat desc_img = img(r);
            kpt->setDescriptor(desc_img, "block_feature");

            // Adding keypoint to frame data.
            frame->addKeypoint(kpt);
        }
    }
}