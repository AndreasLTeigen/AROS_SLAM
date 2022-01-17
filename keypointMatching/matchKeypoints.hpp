#ifndef matchKeypoints_h
#define matchKeypoints_h

#include "../dataStructures/frameData.hpp"

enum class Matcher {brute_force_mono, NONE};

Matcher getMatchingMethod( std::string matching_method );
void matchKeypoints( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2, Matcher matcher_type );

#endif