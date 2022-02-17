#ifndef motionPrior_h
#define motionPrior_h

#include "../dataStructures/frameData.hpp"


class MotionPrior
{
    public:
        MotionPrior(){};
        ~MotionPrior(){};

        virtual void calculate( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2 )=0;
};

std::shared_ptr<MotionPrior> getMotionPrior( std::string motion_prior_method );




class NoneMP : public MotionPrior
{
    public:
        NoneMP(){};
        ~NoneMP(){};

        void calculate( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2 )override;
};


#endif