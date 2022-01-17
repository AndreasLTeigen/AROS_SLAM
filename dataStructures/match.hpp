#ifndef match_h
#define match_h

#include "keypoint.hpp"
#include <shared_mutex>

// Forward declaration for circular dependence
class KeyPoint2;

class Match
{
    private:
        int match_id;
        std::weak_ptr<KeyPoint2> kpt1;
        std::weak_ptr<KeyPoint2> kpt2;
        float confidence = -1;
        float descr_distance;


        // Mutexes
        mutable std::shared_mutex mutex_match_id;
        mutable std::shared_mutex mutex_kpt1;
        mutable std::shared_mutex mutex_kpt2;
        mutable std::shared_mutex mutex_confidence;
        mutable std::shared_mutex mutex_descr_distance;
    
    public:
        Match(std::shared_ptr<KeyPoint2> kpt1,
              std::shared_ptr<KeyPoint2> kpt2,
              float descr_distance,
              int match_id=-1);
        ~Match();

        // Write functions
        void setMatchID(int match_id);
        void setConfidence(float confidence);
        void setDescriptorDistance(float descr_distance);

        // Read functions
        int getMatchID();
        std::shared_ptr<KeyPoint2> getKpt1();
        std::shared_ptr<KeyPoint2> getKpt2();
        std::shared_ptr<KeyPoint2> getConnectingKpt(int connecting_keypoint_frame_nr);
        float getConfidence();
        float getDescrDistance();
};

#endif