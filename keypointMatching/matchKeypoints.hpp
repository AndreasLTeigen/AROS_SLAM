#ifndef matchKeypoints_h
#define matchKeypoints_h

#include <yaml-cpp/yaml.h>

#include "../dataStructures/frameData.hpp"

class Matcher
{

    public:
        // Logging variable
        int num_matches = -1;

        // Analysis toggle
        bool analysis_match_count = true;

        // Analysis file name
        std::string f_match_count = "output/analysis/match_count.txt";


        Matcher();
        ~Matcher(){};

        virtual int matchKeypoints( std::shared_ptr<FrameData> frame1, 
                                    std::shared_ptr<FrameData> frame2 )=0;
        void analysis(  std::shared_ptr<FrameData> frame1,
                                std::shared_ptr<FrameData> frame2 );

        //Logging
        int getCurrMatchNum();
};

std::shared_ptr<Matcher> getMatcher( YAML::Node config );




class NoneMatcher : public Matcher
{
    public:
        NoneMatcher();
        ~NoneMatcher(){};

        int matchKeypoints( std::shared_ptr<FrameData> frame1, 
                            std::shared_ptr<FrameData> frame2 )override;
};

#endif