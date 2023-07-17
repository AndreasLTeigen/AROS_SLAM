#ifndef depthGT_h
#define depthGT_h

#include "../mapPointRegistration.hpp"
#include "../../dataStructures/frameData.hpp"
#include "../../dataStructures/map3D.hpp"

class depthGTMPReg : public MapPointRegistrator
{
    public:
        depthGTMPReg(){};
        ~depthGTMPReg(){};

        int registerMP( std::shared_ptr<FrameData> frame1, 
                        std::shared_ptr<FrameData> frame2, 
                        std::shared_ptr<Map3D> map_3d )override;
};

#endif