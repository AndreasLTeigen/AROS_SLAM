#ifndef mappoint_h
#define mappoint_h

#include "keypoint.hpp"
#include <shared_mutex>

// Forward declaration for circular dependence
class KeyPoint2;

class MapPoint
{
    private:
        int map_point_nr, observation_cnt;
        double X;
        double Y;
        double Z;
        double std_X;
        double std_Y;
        double std_Z;
        cv::Mat mean_view_dir;
        std::map<std::string, cv::Mat> rep_descr;
        std::map<int, std::weak_ptr<KeyPoint2>> observation_keypoints;


        // Mutexes
        mutable std::shared_mutex mutex_coord;
        mutable std::shared_mutex mutex_coord_std;
        mutable std::shared_mutex mutex_mean_view_dir;
        mutable std::shared_mutex mutex_rep_descr;
        mutable std::shared_mutex mutex_observation_keypoints;
        mutable std::shared_mutex mutex_observation_cnt;

    public:
        MapPoint(double X, double Y, double Z, double std_X, double std_Y, double std_Z, 
                    std::shared_ptr<KeyPoint2> kpt1, std::shared_ptr<KeyPoint2> kpt2,
                    cv::Mat T1, cv::Mat T2);
        ~MapPoint();

        // Write functions
        void iterateObservationCounter();
        void setCoordX(double X);
        void setCoordY(double Y);
        void setCoordZ(double Z);
        void setCoordXYZ(cv::Mat XYZ);
        void setSTDX(double std_X);
        void setSTDY(double std_Y);
        void setSTDZ(double std_Z);
        void update3DLocation( cv::Mat XYZ );
        void update3DUncertainty( cv::Mat XYZ_std );
        void updateMeanViewDir(cv::Mat T);
        void updateRepresentativeDescriptor(cv::Mat descriptor, std::string descr_type="orb");
        void addObservation(std::shared_ptr<KeyPoint2> kpt, cv::Mat XYZ, cv::Mat XYZ_std, cv::Mat T, std::string descr_type="orb");
        void addObservationKpt(std::shared_ptr<KeyPoint2> kpt);

        // Read functions
        int getObservationCounter();
        double getCoordX();
        double getCoordY();
        double getCoordZ();
        cv::Mat getCoordXYZ();
        double getSTDX();
        double getSTDY();
        double getSTDZ();
        cv::Mat getMeanViewingDir();
        cv::Mat getRepresentativeDescriptor(std::string descr_type="orb");
        std::weak_ptr<KeyPoint2> getObservationKpt(int kpt_frame_nr);
};


#endif