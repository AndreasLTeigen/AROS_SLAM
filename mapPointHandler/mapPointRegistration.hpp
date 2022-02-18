#ifndef mapPointRegistration_h
#define mapPointRegistration_h

#include "../dataStructures/frameData.hpp"
#include "../dataStructures/map3D.hpp"

class MapPointRegistrator
{
    public:
        MapPointRegistrator(){};
        ~MapPointRegistrator(){};

        virtual void registerMP( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2, std::shared_ptr<Map3D> map_3d )=0;
};

std::shared_ptr<MapPointRegistrator> getMapPointRegistrator( std::string map_point_reg_method );


class NoneMPReg : public MapPointRegistrator
{
    public:
        NoneMPReg(){};
        ~NoneMPReg(){};

        void registerMP( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2, std::shared_ptr<Map3D> map_3d )override;
};

#endif