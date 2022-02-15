#ifndef sequencer_h
#define sequencer_h

#include "opencv2/opencv.hpp"


class Sequencer{
private:
	int frame_buffer_size;
	std::queue<cv::Mat> frame_buffer;
	cv::Point2d mouse_coords;
	int pressed_key;
	int fps;
	std::string file_format;
	bool play_mode;
	int play_speed;
	bool slow_motion;
	bool just_finished;
	bool finished;
	std::string path;
	int current_index;
	int max_index;
	std::vector<std::string> img_seq;
	cv::Mat out;
	cv::Mat img_current;
	float scale;
	cv::Point2d origin_offset;
	int width;
	int height;
	cv::Mat control_panel;
	bool recording;
	cv::VideoWriter recorder;
	std::string record_dst_path;
	std::string record_name;


public:
	Sequencer(std::string folder_path, int frame_buffer_size, std::string file_format = "png", bool recording=false, std::string record_dst_path=nullptr, std::string record_name=nullptr, int fps = 1);
	~Sequencer();

	int getFrameBufferSize();
	bool hasNextFrame();
	cv::Mat getCurrentFrame();
	cv::Mat getNextFrame();
	std::string getCurrentPath();
	std::string getCurrentName();
	int getCurrentIndex();
	void setCurrentIndex(int index);
	cv::Point2d getMouseCoords();
	int getPressedKey();
	bool isJustFinished();
	bool isFinished();
	void preLoadFrameBufferFull();
	void preLoadFrameBufferFrame(cv::Mat frame);
	void frameBufferPush(cv::Mat img);
	cv::Mat frameBufferGetNewest();
	cv::Mat frameBufferGetOldest();
	void visualizeImage(cv::Mat &img);
	void visualizeImageNext(cv::Mat &img);
	void recordImage(cv::Mat &img);
	void show();
	void evalMouse(std::string event, int x, int y);
	void evalKey(char key);
	void iterateToNewFrame();
	void setupGui();
};

void checkFrameIntegrity(cv::Mat frame);

#endif