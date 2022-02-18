#ifndef linearInclusiveMPReg_h
#define linearInclusiveMPReg_h

#include "../mapPointRegistration.hpp"
#include "../../dataStructures/frameData.hpp"
#include "../../dataStructures/map3D.hpp"

void linearInclusiveMPReg( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2, std::shared_ptr<Map3D> map_3d );


class LinIncMPReg : public MapPointRegistrator
{
    public:
        LinIncMPReg(){};
        ~LinIncMPReg(){};

        void registerMP( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2, std::shared_ptr<Map3D> map_3d )override;
};

#endif