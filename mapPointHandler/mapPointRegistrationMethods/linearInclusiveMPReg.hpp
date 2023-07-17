#ifndef linearInclusiveMPReg_h
#define linearInclusiveMPReg_h

#include "../mapPointRegistration.hpp"
#include "../../dataStructures/frameData.hpp"
#include "../../dataStructures/map3D.hpp"

void linearInclusiveMPReg(  std::shared_ptr<FrameData> frame1, 
                            std::shared_ptr<FrameData> frame2, 
                            std::shared_ptr<Map3D> map_3d );


class LinIncMPReg : public MapPointRegistrator
{
    public:
        LinIncMPReg(){};
        ~LinIncMPReg(){};

        int registerMP( std::shared_ptr<FrameData> frame1, 
                        std::shared_ptr<FrameData> frame2, 
                        std::shared_ptr<Map3D> map_3d )override;
        int removeInvalidPoints( cv::Mat& X_c );
        void isPointBehindCamera( cv::Mat& X_c, cv::Mat& ret );
};

#endif