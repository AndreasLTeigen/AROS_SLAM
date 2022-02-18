#ifndef mapPointCulling_h
#define mapPointCulling_h

class MapPointCuller
{
    public:
        MapPointCuller(){};
        ~MapPointCuller(){};

        virtual void cullMP()=0;
};

std::shared_ptr<MapPointCuller> getMapPointCuller( std::string map_point_cull_method );

class NoneMPCull : public MapPointCuller
{
    public:
        NoneMPCull(){};
        ~NoneMPCull(){};

        void cullMP()override;
};

#endif