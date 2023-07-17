#ifndef map3D_h
#define map3D_h

#include "mapPoint.hpp"

class Map3D
{
    private:
        std::vector<std::shared_ptr<MapPoint>> map_points;

        // Mutexes
        mutable std::shared_mutex mutex_map_points;
    
    public:
        Map3D();
        ~Map3D();

        // Write functions
        void addMapPoint(std::shared_ptr<MapPoint> map_point);
        void createMapPoint(cv::Mat XYZ, cv::Mat XYZ_std, std::shared_ptr<KeyPoint2> kpt1, cv::Mat T1);
        void removeMapPoint(int idx);
        void updateMap(std::shared_ptr<KeyPoint2> kpt1, std::shared_ptr<KeyPoint2> kpt2, cv::Mat T1, cv::Mat T2, cv::Mat XYZ1, cv::Mat XYZ_std);
        void batchUpdateMap(std::vector<std::shared_ptr<KeyPoint2>>& kpts1, std::vector<std::shared_ptr<KeyPoint2>>& kpts2, cv::Mat T1, cv::Mat T2, cv::Mat XYZ, cv::Mat XYZ_std);
        void resetMap();

        // Read functions
        int getNumMapPoints();
        std::shared_ptr<MapPoint> getMapPoint(int idx);
        std::vector<std::shared_ptr<MapPoint>> getAllMapPoints();
        cv::Mat compileMapPointLocs();
        std::vector<cv::Point3f> compileCVPoints3f();
        std::vector<cv::Point3d> compileCVPoints3d();

        // Static functions
        static void calculateReprojectionError(cv::Mat uv1, cv::Mat uv2, cv::Mat K1, cv::Mat K2, cv::Mat T1, cv::Mat T2, cv::Mat reproj_error1, cv::Mat reproj_error2);
        static cv::Mat calculate3DUncertainty(cv::Mat XYZ, cv::Mat uv1, cv::Mat uv2, cv::Mat K1, cv::Mat K2, cv::Mat T1, cv::Mat T2);

};

#endif