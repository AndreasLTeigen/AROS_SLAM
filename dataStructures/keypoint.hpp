#ifndef keypoint_h
#define keypoint_h

#include "match.hpp"
#include "mapPoint.hpp"
#include <map>
#include <shared_mutex>
#include <opencv2/opencv.hpp>

// Forward declaration for circular dependence
class Match;
class MapPoint;

class KeyPoint2
{
    private:
        int kpt_id;             // Unique per frame identifier
        double x;
        double y;
        double angle;           
        int octave;             // Pyramid layer from which keypoint was extracted
        double response;        //  response by which the most strong keypoints have been selected
        double size;            //  diameter of meaningfull keypoint neighborhood
        int observation_frame_nr;
        std::map<std::string, cv::Mat> descriptors;
        std::map<int, std::vector<std::shared_ptr<Match>>> matches;

        std::shared_ptr<MapPoint> map_point;

        // Mutexes
        mutable std::shared_mutex mutex_kpt_id;
        mutable std::shared_mutex mutex_coord;
        mutable std::shared_mutex mutex_angle;
        mutable std::shared_mutex mutex_octave;
        mutable std::shared_mutex mutex_response;
        mutable std::shared_mutex mutex_size;
        mutable std::shared_mutex mutex_observation_frame_nr;
        mutable std::shared_mutex mutex_descriptors;
        mutable std::shared_mutex mutex_matches_map;
        mutable std::shared_mutex mutex_map_point;


    public:
        KeyPoint2( int kpt_id, cv::KeyPoint kpt, int observation_frame_nr );
        KeyPoint2( int kpt_id, cv::KeyPoint kpt, int observation_frame_nr, cv::Mat descr, std::string descr_type="orb" );
        KeyPoint2( int kpt_id, cv::Mat xy1, int observation_frame_nr, double angle=-1, int octave=-1, double response=-1, double size=-1 );
        ~KeyPoint2();

        // Write funtions
        void setKptID(int kpt_id);
        void setCoordx(double x);
        void setCoordy(double y);
        void setAngle(double angle);
        void setMapPointID(int mapPoint_id); //TODO: Should be able to remove this function
        void setOctave(int octave);
        void setResponse(int response);
        void setSize(int size);
        void setObservationFrameNr(int observation_frame_nr);
        void setDescriptor(cv::Mat descr, std::string descr_type="orb");
        void setMapPoint(std::shared_ptr<MapPoint> map_point);
        void addMatch(std::shared_ptr<Match> match, int matched_frame_nr);
        void removeMatchReference(int matched_frame_nr, std::shared_ptr<Match> remove_match);
        void removeAllMatchReferences(int matched_frame_nr);
        void removeAllMatches(int matched_frame_nr);
        void orderMatchesByConfidence(int matched_frame_nr);

        // Read funcitons
        int getKptId();
        double getCoordX();
        double getCoordY();
        cv::Mat getLoc();
        int getOctave();
        int getObservationFrameNr();
        double getAngle();
        double getResponse();
        double getSize();
        bool isDescriptor(std::string descr_type);
        cv::Mat getDescriptor(std::string descr_type="orb");
        std::shared_ptr<MapPoint> getMapPoint();
        std::vector<std::shared_ptr<Match>> getMatches(int matched_frame_nr);
        //std::vector<std::shared_ptr<KeyPoint2>> getBestMatchedKpt( int matched_frame_nr ) TODO: Implement this function.
        std::shared_ptr<Match> getHighestConfidenceMatch(int matched_frame_nr);
        cv::Point compileCV2DPoint();   //TODO: Make this into a cv::Point2f return value
        cv::Mat compileHomogeneousCV2DPoint();
        cv::KeyPoint compileCVKeyPoint();

        // Static funtions
        static double calculateKeypointDistance(std::shared_ptr<KeyPoint2> kpt1, std::shared_ptr<KeyPoint2> kpt2);
        static void drawEnchancedKeyPoint( cv::Mat &canvas, cv::Mat &img, std::shared_ptr<KeyPoint2> kpt, cv::Point loc_canvas, cv::Size size, cv::Mat F_matrix, std::shared_ptr<KeyPoint2> matched_kpt=nullptr );
        static void drawKptHeatMapAnalysis( cv::Mat &canvas, cv::Mat &img, std::shared_ptr<KeyPoint2> kpt, cv::Point loc_canvas, cv::Size size, cv::Mat F_matrix, std::shared_ptr<KeyPoint2> matched_kpt, cv::Mat heat_map );

};

#endif