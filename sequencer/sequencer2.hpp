#ifndef sequencer2_h
#define sequencer2_h

class Sequencer2
{
    private:
        bool webcam = false;
        int curr_idx;
        int playback_mode;
        std::string video_path;
        cv::VideoCapture video_cap;
        std::vector<cv::Mat> frame_buffer;

    public:
        Sequencer2(std::string source, int playback_mode=1, int start_idx=0);
        ~Sequencer2();
        int getCurrentIdx();
        std::string getCurrentImgName();    //Change to getCurrentFrameName
        cv::Mat getCurrentImg();            //Change to getCurrentFrame
        void visualizeImage(cv::Mat& img);
        bool hasNextFrame();
        void iterateToNewFrame();
        bool isFinished();
};

#endif


/*
Init:
    Arguments:
        int Speed mode (1,2)
        int num_cam
        vector<string> source
        int fps
        bool reverse
        int frame_buffer_size
*/

// Functionality

    /*
    Multi Camera:
        Load multiple camera sources at once.
        Return multiple camera sources at once.
    */

    //If video source is stored video

        /*
        Speed Modes: 
            Fast(1): Quick as possible.
            Normal(2): Max(Fps, Fast).
        */

        /*
            If pause, allow manual stepping
        */

        /*
        Reverse:
            Run entire sequence in reverse.
        */

        /*
        Skip Frames: 
            X: Skip x amount of frames between every loaded frame, show them at <Speed Modes> speed.
        */

        /*
        Multithreading frame buffer:
            Faster loading of images.
        */

    //If video source is webcam
        // Load next image