#include <map>
#include <math.h>
#include <memory>
#include <vector>
#include <algorithm>
#include <shared_mutex>
#include <opencv2/opencv.hpp>

#include "keypoint.hpp"
#include "frameData.hpp"
#include "../util/util.hpp"

using std::map;
using std::vector;
using std::shared_ptr;
using std::unique_ptr;
using cv::Mat;



KeyPoint2::KeyPoint2( int kpt_id, cv::Point pt, int observation_frame_nr )
{
    this->setKptID(kpt_id);
    this->setCoordx(pt.x);
    this->setCoordy(pt.y);
    this->setObservationFrameNr(observation_frame_nr);
    this->setColor(0);
    this->setSize(0);
    this->setAngle(-1);
    this->setResponse(0);
    this->setOctave(0);
    this->setMapPoint(nullptr);
}

KeyPoint2::KeyPoint2( int kpt_id, cv::KeyPoint kpt, int observation_frame_nr )
{
    this->setKptID(kpt_id);
    this->setCoordx(kpt.pt.x);
    this->setCoordy(kpt.pt.y);
    this->setObservationFrameNr(observation_frame_nr);
    this->setColor(0);
    this->setSize(kpt.size);
    this->setAngle(kpt.angle);
    this->setResponse(kpt.response);
    this->setOctave(kpt.octave);
    this->setMapPoint(nullptr);
}

KeyPoint2::KeyPoint2( int kpt_id, cv::KeyPoint kpt, int observation_frame_nr, Mat descr, std::string descr_type )
{
    this->setKptID(kpt_id);
    this->setCoordx(kpt.pt.x);
    this->setCoordy(kpt.pt.y);
    this->setObservationFrameNr(observation_frame_nr);
    this->setColor(0);
    this->setSize(kpt.size);
    this->setAngle(kpt.angle);
    this->setResponse(kpt.response);
    this->setOctave(kpt.octave);
    this->setDescriptor(descr, descr_type);
    this->setMapPoint(nullptr);
}

KeyPoint2::KeyPoint2( int kpt_id, int x, int y, int observation_frame_nr, double angle, int octave, double response, double size)
{
    this->setKptID(kpt_id);
    this->setCoordx(x);
    this->setCoordy(y);
    this->setObservationFrameNr(observation_frame_nr);
    this->setColor(0);
    this->setSize(size);
    this->setAngle(angle);
    this->setResponse(response);
    this->setOctave(octave);
    this->setMapPoint(nullptr);
}

KeyPoint2::KeyPoint2( int kpt_id, cv::Mat xy1, int observation_frame_nr, double angle, int octave, double response, double size)
{
    /*
    Arguments:
        xy1:    Homogeneous pixel coordinates [shape 3 x 1].
    */

    this->setKptID(kpt_id);
    this->setCoordx(xy1.at<double>(0,0));
    this->setCoordy(xy1.at<double>(1,0));
    this->setObservationFrameNr(observation_frame_nr);
    this->setColor(0);
    this->setSize(size);
    this->setAngle(angle);
    this->setResponse(response);
    this->setOctave(octave);
    this->setMapPoint(nullptr);
}

KeyPoint2::~KeyPoint2()
{
    //TODO: Implement destructor
    //std::cout << "Keypoint Destroyed" << std::endl;
}

void KeyPoint2::setValidFlag(bool valid)
{
    std::unique_lock lock(this->mutex_valid);
    this->valid = valid;
}

void KeyPoint2::setKptID(int kpt_id)
{
    std::unique_lock lock(this->mutex_kpt_id);
    this->kpt_id = kpt_id;
}

void KeyPoint2::setCoordx(double x)
{
    std::unique_lock lock(this->mutex_coord);
    this->x = x;
}

void KeyPoint2::setCoordy(double y)
{
    std::unique_lock lock(this->mutex_coord);
    this->y = y;
}

void KeyPoint2::setAngle(double angle)
{
    std::unique_lock lock(this->mutex_angle);
    this->angle = angle;
}

void KeyPoint2::setOctave(int octave)
{
    std::unique_lock lock(this->mutex_octave);
    this->octave = octave;
}

void KeyPoint2::setResponse(int response)
{
    std::unique_lock lock(this->mutex_response);
    this->response = response;
}

void KeyPoint2::setSize(int size)
{
    std::unique_lock lock(this->mutex_size);
    this->size = size;
}

void KeyPoint2::setObservationFrameNr(int observation_frame_nr)
{
    std::unique_lock lock(this->mutex_observation_frame_nr);
    this->observation_frame_nr = observation_frame_nr;
}

void KeyPoint2::setColor( uint8_t color )
{
    std::unique_lock lock(this->mutex_color);
    this->color = color;
}

void KeyPoint2::setDescriptor(Mat descr, std::string descr_type)
{
    std::unique_lock lock(this->mutex_descriptors);
    this->descriptors[descr_type] = descr;
}

void KeyPoint2::setMapPoint(std::shared_ptr<MapPoint> map_point)
{
    std::unique_lock lock(this->mutex_map_point);
    this->map_point = map_point;
}

void KeyPoint2::addMatch(shared_ptr<Match> match, int matched_frame_nr)
{
    std::unique_lock lock(this->mutex_matches_map);
    this->matches[matched_frame_nr].push_back(match);
}

void KeyPoint2::removeMatchReference(int matched_frame_nr, std::shared_ptr<Match> removing_match)
{
    // Removes reference to keypoint's specific match <removing_match> with frame nr. <matched_frame_nr>
    // WARNING: DOES NOT REMOVE THE REFERENCES TO THE MATCH FOR THE MATCHED KEYPOINT
    std::unique_lock lock(this->mutex_matches_map);

    std::vector<std::shared_ptr<Match>> matches = this->matches[matched_frame_nr];
    std::remove(matches.begin(), matches.end(), removing_match);
}

void KeyPoint2::removeAllMatchReferences(int matched_frame_nr)
{
    // Removes references to all keypoint's matches with frame nr. <matched_frame_nr>
    // WARNING: DOES NOT REMOVE THE REFERENCES TO THE MATCH FOR THE MATCHED KEYPOINT
    std::unique_lock lock(this->mutex_matches_map);

    this->matches.erase( matched_frame_nr ); 
}

void KeyPoint2::removeAllMatches(int matched_frame_nr)
{
    //TODO: Make this into a static function.
    std::shared_ptr<KeyPoint2> temp_kpt;

    std::unique_lock lock(this->mutex_matches_map);
    for ( shared_ptr<Match> match : this->matches[matched_frame_nr] )
    {
        temp_kpt = match->getConnectingKpt(matched_frame_nr);
        temp_kpt->removeAllMatchReferences(this->getObservationFrameNr());
    }
    this->matches.erase( matched_frame_nr );
}

void KeyPoint2::orderMatchesByConfidence(int matched_frame_nr)
{
    //TODO: MAKE THIS FUNCTION, SOME FUNCTION RELIES ON AN ORDERED KEYPOINT LIST
}

bool KeyPoint2::isValid()
{
    std::shared_lock lock(this->mutex_valid);
    return this->valid;
}

int KeyPoint2::getKptId()
{
    std::shared_lock lock(this->mutex_kpt_id);
    return this->kpt_id;
}

double KeyPoint2::getCoordX()
{
    std::shared_lock lock(this->mutex_coord);
    return this->x;
}

double KeyPoint2::getCoordY()
{
    std::shared_lock lock(this->mutex_coord);
    return this->y;
}

cv::Mat KeyPoint2::getLoc()
{
    /*
    Returns:
        loc:    Homogeneous location of keypoint (x, y, 1).T [3 x 1].
    */
    cv::Mat loc = (cv::Mat_<double>(3,1) << this->getCoordX(), this->getCoordY(), 1);
    return loc;
}

int KeyPoint2::getOctave()
{
    std::shared_lock lock(this->mutex_octave);
    return this->octave;
}

int KeyPoint2::getObservationFrameNr()
{
    std::shared_lock lock(this->mutex_observation_frame_nr);
    return this->observation_frame_nr;
}

uint8_t KeyPoint2::getColor()
{
    std::shared_lock lock(this->mutex_color);
    return this->color;
}

double KeyPoint2::getAngle()
{
    std::shared_lock lock(this->mutex_angle);
    return this->angle;
}

double KeyPoint2::getResponse()
{
    std::shared_lock lock(this->mutex_response);
    return this->response;
}

double KeyPoint2::getSize()
{
    std::shared_lock lock(this->mutex_size);
    return this->size;
}

bool KeyPoint2::isDescriptor(std::string descr_type)
{
    std::shared_lock lock(mutex_descriptors);
    if ( this->descriptors.find(descr_type) == this->descriptors.end() )
    {
        return false;
    }
    else
    {
        return true;
    }
}

Mat KeyPoint2::getDescriptor(std::string descr_type)
{
    std::shared_lock lock(mutex_descriptors);
    return this->descriptors[descr_type];
}

std::shared_ptr<MapPoint> KeyPoint2::getMapPoint()
{
    std::shared_lock lock(this->mutex_map_point);
    return this->map_point;
}

vector<shared_ptr<Match>> KeyPoint2::getMatches(int matched_frame_nr)
{
    std::shared_lock lock(this->mutex_matches_map);
    return this->matches[matched_frame_nr];
}

shared_ptr<Match> KeyPoint2::getHighestConfidenceMatch(int matched_frame_nr)
{
    // Assumes a sorted list
    if (this->getMatches(matched_frame_nr).size() == 0)
    {
        return nullptr;
    }
    else
    {
        return this->getMatches(matched_frame_nr)[0];
    }
}

cv::Point KeyPoint2::compileCV2DPoint()
{
    unique_ptr<cv::Point> point = unique_ptr<cv::Point>(new cv::Point(this->getCoordX(), this->getCoordY()));
    return *point;
}

Mat KeyPoint2::compileHomogeneousCV2DPoint()
{
    return (cv::Mat_<double>(3,1) << this->getCoordX(), this->getCoordY(), 1);
}

cv::KeyPoint KeyPoint2::compileCVKeyPoint()
{
    cv::KeyPoint kpt_cv = cv::KeyPoint(this->getCoordX(), this->getCoordY(),
                                        this->getSize(), this->getAngle(),
                                        this->getResponse(), this->getOctave());
    return kpt_cv;
}

// Static functions
double KeyPoint2::calculateKeypointDistance(shared_ptr<KeyPoint2> kpt1, shared_ptr<KeyPoint2> kpt2)
{
    return sqrt((kpt1->getCoordX() - kpt2->getCoordX())*(kpt1->getCoordX() - kpt2->getCoordX()) + (kpt1->getCoordY() - kpt2->getCoordY())*(kpt1->getCoordY() - kpt2->getCoordY()));
}

void KeyPoint2::drawEnhancedKeyPoint( cv::Mat &canvas, cv::Mat &img, std::shared_ptr<KeyPoint2> kpt, 
                                        cv::Point loc_canvas, cv::Size size, cv::Mat F_matrix,
                                         shared_ptr<KeyPoint2> matched_kpt )
{
    int reg_w, reg_h, top, bottom, left, right;
    cv::Mat img_sec, roi;

    // Extracting and enhancing keypoint image section
    reg_w = kpt->getSize();
    reg_h = reg_w;
    
    top = std::max(int(kpt->getCoordY() - reg_h/2), 0);
    left = std::max(int(kpt->getCoordX() - reg_w/2), 0);
    reg_h = std::min(reg_h, img.rows - int(kpt->getCoordY()));
    reg_w = std::min(reg_w, img.cols - int(kpt->getCoordX()));

    img_sec = img(cv::Rect(left, top, reg_w, reg_h)).clone();
    cv::resize(img_sec, img_sec, size);

    // Draw keypoint center
    cv::Scalar blue(255, 0, 0);
    int cross_hair_size = 20;
    int kpt_x = int( (kpt->getCoordX() - left)*size.width/reg_w );
    int kpt_y = int( (kpt->getCoordY() - top)*size.height/reg_h );
    cv::line(img_sec,   cv::Point(kpt_x - cross_hair_size, kpt_y),
                        cv::Point(kpt_x + cross_hair_size, kpt_y),
                        blue);
    cv::line(img_sec,   cv::Point(kpt_x, kpt_y - cross_hair_size),
                        cv::Point(kpt_x, kpt_y + cross_hair_size),
                        blue);
    
    // Draw epipolar line
    if ( matched_kpt != nullptr )
    {
        //std::cout << kpt->getLoc().t() << std::endl;
        //std::cout << F_matrix << std::endl;
        //std::cout << matched_kpt->getLoc() << std::endl;
        // std::cout << (kpt->getLoc()).t() * F_matrix.t() * matched_kpt->getLoc() << std::endl;

        cv::Scalar red(0, 0, 255);
        //std::vector<cv::Point3f> epiline;
        //cv::Point matched_point = matched_kpt->compileCV2DPoint();
        //vector<cv::Point> point2{matched_point};
        //cv::computeCorrespondEpilines(point2, 1, F_matrix.t(), epiline);

        //double a = - epiline[0].x / epiline[0].y;
        //double b = - epiline[0].z / epiline[0].y;

        cv::Mat epiline = F_matrix * matched_kpt->getLoc();
        // std::cout << epiline << std::endl;
        // std::cout << kpt->getLoc().t() * epiline << std::endl;
        double a = - epiline.at<double>(0,0) / epiline.at<double>(1,0);
        double b = - epiline.at<double>(2,0) / epiline.at<double>(1,0);
        // std::cout << kpt->getCoordY() << " = " << kpt->getCoordX() * a + b << std::endl; 


        // Checking intersects in all vertices of the image patch
        std::vector<cv::Point> points;

        // Left intersection
        if ( (a * left + b) > top && (a * left + b) < top + reg_h )
        {
            points.push_back(cv::Point(left, (a * left + b)));
        }
        // Top instersection
        if ( (top - b)/a > left && (top - b)/a < left + reg_w )
        {
            points.push_back(cv::Point((top - b)/a, top));
        }

        // Right intersection
        if ( (a * (left+reg_w) + b) > top && (a * (left+reg_w) + b) < top + reg_h )
        {
            points.push_back(cv::Point((left+reg_w), (a * (left+reg_w) + b)));
        }
        // Bottom intersection
        if ( ((top + reg_h) - b)/a > left && ((top + reg_h) - b)/a < left + reg_w )
        {
            points.push_back(cv::Point(((top + reg_h) - b)/a, (top + reg_h)));
        }

        if (points.size() == 2)
        {
            points[0].x -= left;
            points[0].y -= top;
            points[1].x -= left;
            points[1].y -= top;

            points[0].x *= (size.width/reg_w);
            points[0].y *= (size.height/reg_h);
            points[1].x *= 2*(size.width/reg_w);
            points[1].y *= 2*(size.height/reg_h);

            //std::cout << points[0] << points[1] << std::endl;
            cv::line(img_sec, points[0], points[1], red);
        }
        else if (points.size() > 2)
        {
            std::cout << "WARNING: LINE INTERSECTS SQUARE MORE THAN 2 TIMES" << std::endl;
        }

    }

    // Copying enhanced keypoint into canvas
    roi = canvas(cv::Rect(loc_canvas.x, loc_canvas.y, img_sec.cols, img_sec.rows));
    // img_sec.copyTo( canvas(cv::Rect(loc_canvas.x, loc_canvas.y, img_sec.cols, img_sec.rows)) );
    img_sec.copyTo(roi);
}

// Function specific to GJET
// TODO: MOVE TO GJET IMPLEMENTATION
void KeyPoint2::drawKptHeatMapAnalysis( cv::Mat &canvas, cv::Mat &img, std::shared_ptr<KeyPoint2> kpt, 
                                        cv::Point loc_canvas, cv::Size size, cv::Mat F_matrix,
                                         shared_ptr<KeyPoint2> matched_kpt, cv::Mat heat_map, int it, bool updated,
                                         bool show_hm )
{
    if (!heat_map.empty())
    {
        int reg_w, reg_h, top, bottom, left, right;
        cv::Mat img_sec, roi, uv, v_k_opt;

        // Extracting and enhancing keypoint image section
        reg_w = kpt->getSize();
        reg_h = reg_w;
        
        uv = kpt->getDescriptor("loc_from_log" + std::to_string(it));

        top = std::max(int(uv.at<double>(1,0) - reg_h/2), 0);
        left = std::max(int(uv.at<double>(0,0) - reg_w/2), 0);
        reg_h = std::min(reg_h, img.rows - int(uv.at<double>(1,0)));
        reg_w = std::min(reg_w, img.cols - int(uv.at<double>(0,0)));

        img_sec = img(cv::Rect(left, top, reg_w, reg_h)).clone();

        // std::cout << kpt << std::endl;
        // std::cout << kpt->getLoc() << std::endl;
        // std::cout << kpt->getDescriptor("init") << std::endl;

        // Adding heatmap overlay of hamming distance
        if (show_hm)
        {
            cv::Mat heatmap_img, heat_map_img_sec;
            cv::Mat heat_map1 = 255 - heat_map;
            heat_map_img_sec = img_sec(cv::Rect(int(img_sec.cols/2) - int(heat_map.cols/2), 
                                                int(img_sec.rows/2) - int(heat_map.rows/2), 
                                                heat_map.cols, heat_map.rows)).clone();

            heat_map1.convertTo(heat_map1, CV_8UC1);
            cv::applyColorMap(heat_map1, heatmap_img, cv::COLORMAP_JET);
            cv::addWeighted(heatmap_img, 0.5, heat_map_img_sec, 0.5, 0, heat_map_img_sec);
            heat_map_img_sec.copyTo( img_sec(cv::Rect(int(img_sec.cols/2) - int(heat_map.cols/2), 
                                                    int(img_sec.rows/2) - int(heat_map.rows/2), 
                                                    heat_map.cols, heat_map.rows)) );
        }

        cv::resize(img_sec, img_sec, size);


        v_k_opt = kpt->getDescriptor("v_k_opt_log" + std::to_string(it));

        // Draw keypoint center
        cv::Scalar blue(255, 0, 0);
        int cross_hair_size = 20;
        int kpt_x, kpt_y;

        if (updated)
        {
            // kpt_x = int( (uv.at<double>(0,0) - left)*size.width/reg_w ); // C1
            // kpt_y = int( (uv.at<double>(1,0) - top)*size.height/reg_h ); // C2
            kpt_x = int( (uv.at<double>(0,0) + v_k_opt.at<double>(0,0) - left)*size.width/reg_w); // C1
            kpt_y = int( (uv.at<double>(1,0) + v_k_opt.at<double>(1,0) - top)*size.height/reg_h); // C2
            //kpt_x = int((kpt->getCoordX() - left)*size.width/reg_w);
            //kpt_y = int((kpt->getCoordY() - top)*size.height/reg_h);
        }
        else
        {
            // kpt_x = int( (uv.at<double>(0,0) - v_k_opt.at<double>(0,0) - left)*size.width/reg_w); // C3
            // kpt_y = int( (uv.at<double>(1,0) - v_k_opt.at<double>(1,0) - top)*size.height/reg_h); // C4
            kpt_x = int( (uv.at<double>(0,0) - left)*size.width/reg_w ); // C3
            kpt_y = int( (uv.at<double>(1,0) - top)*size.height/reg_h ); // C4
        }
        cv::line(img_sec,   cv::Point(kpt_x - cross_hair_size, kpt_y),
                            cv::Point(kpt_x + cross_hair_size, kpt_y),
                            blue);
        cv::line(img_sec,   cv::Point(kpt_x, kpt_y - cross_hair_size),
                            cv::Point(kpt_x, kpt_y + cross_hair_size),
                            blue);



        /*
        std::cout <<"#############################################\nIteration: " << it << std::endl;
        std::cout << "A: " << kpt->getDescriptor("quad_fit_log" + std::to_string(it)) << std::endl;
        std::cout << "kpt1:\n" << kpt->getDescriptor("loc_from_log" + std::to_string(it)) << std::endl;
        std::cout << "kpt2:\n" << matched_kpt->getLoc() << std::endl;
        std::cout << "F_matrix:\n" << kpt->getDescriptor("F_matrix_log" + std::to_string(it)) << std::endl;
        //std::cout << "Opt point:\n" << kpt->getDescriptor("v_k_opt_log" + std::to_string(it)) << std::endl;
        std::cout << "Opt point:\n" << kpt->getDescriptor("loc_from_log" + std::to_string(it)).rowRange(0,2) + kpt->getDescriptor("v_k_opt_log" + std::to_string(it)) << std::endl;
        */
        
        // Draw epipolar line
        if ( matched_kpt != nullptr )
        {
            cv::Scalar red(0, 0, 255);

            /*
            std::vector<cv::Point3f> epiline1;
            cv::Point matched_point = matched_kpt->compileCV2DPoint();
            vector<cv::Point> point2{matched_point};

            cv::computeCorrespondEpilines(point2, 1, F_matrix, epiline1);
            double a = - epiline1[0].x / epiline1[0].y;
            double b = - epiline1[0].z / epiline1[0].y;
            std::cout << "a: " << a << ", b: " << b << std::endl; 
            */


            cv::Mat epiline, x2_k;
            x2_k = matched_kpt->getLoc();
            epiline = F_matrix * x2_k;
            double a = -epiline.at<double>(0,0) / epiline.at<double>(1,0);
            double b = -epiline.at<double>(2,0) / epiline.at<double>(1,0);
            //std::cout << "a: " << a << ", b: " << b << std::endl; 

            // std::cout << "Epiline: " << kpt->getLoc().t() * epiline << std::endl;


            // Checking intersects in all vertices of the image patch
            std::vector<cv::Point> points;

            // Left intersection
            if ( (a * left + b) > top && (a * left + b) < top + reg_h )
            {
                //points.push_back(cv::Point(left, (a * left + b)));
                points.push_back(cv::Point(0, ((a * left + b)-top)*(double(size.height)/reg_h)));
            }
            // Top instersection
            if ( (top - b)/a > left && (top - b)/a < left + reg_w )
            {
                //points.push_back(cv::Point((top - b)/a, top));
                points.push_back(cv::Point(((top - b)/a - left)*(double(size.width)/reg_w), 0));
            }
            // Right intersection
            if ( (a * (left+reg_w) + b) > top && (a * (left+reg_w) + b) < top + reg_h )
            {
                points.push_back(cv::Point(size.width, (a * (left+reg_w) + b - top)*(double(size.height)/reg_h)));
            }
            // Bottom intersection
            if ( ((top + reg_h) - b)/a > left && ((top + reg_h) - b)/a < left + reg_w )
            {
                points.push_back(cv::Point((((top + reg_h) - b)/a - left)*(double(size.width)/reg_w), size.height));
            }


            if (points.size() == 2)
            {
                cv::line(img_sec, points[0], points[1], red);
            }
            else if (points.size() > 2)
            {
                std::cout << "WARNING: LINE INTERSECTS SQUARE MORE THAN 2 TIMES" << std::endl;
            }

        }

        // Copying enhanced keypoint into canvas
        roi = canvas(cv::Rect(loc_canvas.x, loc_canvas.y, img_sec.cols, img_sec.rows));
        img_sec.copyTo( canvas(cv::Rect(loc_canvas.x, loc_canvas.y, img_sec.cols, img_sec.rows)) );

        //cv::imshow("KeyPoint Analysis", img_sec);
        //cv::waitKey(0);
    }
}