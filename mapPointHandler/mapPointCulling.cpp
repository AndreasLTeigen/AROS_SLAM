#include <string>
#include <iostream>

#include "mapPointCulling.hpp"

PointCull3D get3DPointCullingMethod(std::string point_cull_3D_method)
{
    if ( point_cull_3D_method == "OoW" )
    {
        // All points that are not visible any frame in the window is culled
        return PointCull3D::OoW;
    }
    else
    {
        std::cout << "ERROR: 3D POINT CULLING METHOD NOT FOUND" << std::endl;
        return PointCull3D::NONE;
    }
}