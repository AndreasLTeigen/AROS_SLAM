#include <string>
#include <memory>
#include <iostream>

#include "mapPointCulling.hpp"


std::shared_ptr<MapPointCuller> getMapPointCuller( std::string map_point_cull_method )
{
    if ( map_point_cull_method == "OoW" )
    {
        std::cerr << "ERROR: MAP POINT CULLING METHOD: 'OoW' NOT IMPLEMENTED" << std::endl;
        return nullptr;
    }
    else
    {
        std::cerr << "ERROR: MAP POINT CULLING METHOD NOT FOUND" << std::endl;
        return nullptr;
    }
}


void NoneMPCull::cullMP()
{
    std::cerr << "ERROR: MAP POINT REGISTRATION ALGORITHM NOT IMPLEMENTED" << std::endl;
}