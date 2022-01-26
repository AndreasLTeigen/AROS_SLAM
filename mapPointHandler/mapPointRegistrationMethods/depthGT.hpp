#ifndef depthGT_h
#define depthGT_h

#include "../../dataStructures/frameData.hpp"
#include "../../dataStructures/map3D.hpp"

void depthGTMPReg( std::shared_ptr<FrameData> frame1, std::shared_ptr<Map3D> map_3d );

#endif