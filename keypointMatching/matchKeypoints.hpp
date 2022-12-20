#ifndef matchKeypoints_h
#define matchKeypoints_h

#include "../dataStructures/frameData.hpp"

class Matcher
{

    public:
        // Logging parameters
        bool is_logging = true;
        int num_match_curr = -1;


        Matcher(){};
        ~Matcher(){};

        virtual void matchKeypoints( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2 )=0;

        //Logging
        int getCurrMatchNum();
};

std::shared_ptr<Matcher> getMatcher( std::string matching_method );




class NoneMatcher : public Matcher
{
    public:
        NoneMatcher(){};
        ~NoneMatcher(){};

        void matchKeypoints( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2 )override;
};

#endif