#ifndef mapPointRegistration_h
#define mapPointRegistration_h

#include "../dataStructures/frameData.hpp"
#include "../dataStructures/map3D.hpp"

enum class PointReg3D {ALL, DEPTH_GT, NONE};

PointReg3D get3DPointRegistrationMethod(std::string point_reg_3D_method);

void register3DPoints( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2, std::shared_ptr<Map3D> map_3d, PointReg3D point_reg_3D );

#endif