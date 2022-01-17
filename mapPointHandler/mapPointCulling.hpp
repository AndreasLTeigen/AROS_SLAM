#ifndef mapPointCulling_h
#define mapPointCulling_h

// OoW - Out of Window
enum class PointCull3D {OoW, NONE};

PointCull3D get3DPointCullingMethod(std::string point_cull_3D_method);

#endif