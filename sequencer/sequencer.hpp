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
	bool has_next_frame();
	cv::Mat get_current_frame();
	cv::Mat get_next_frame();
	std::string get_current_path();
	std::string get_current_name();
	int get_current_index();
	void set_current_index(int index);
	cv::Point2d get_mouse_coords();
	int get_pressed_key();
	bool is_just_finished();
	bool is_finished();
	void pre_load_frame_buffer_full();
	void pre_load_frame_buffer_frame(cv::Mat frame);
	void frame_buffer_push(cv::Mat img);
	cv::Mat frame_buffer_get_newest();
	cv::Mat frame_buffer_get_oldest();
	void visualize_image(cv::Mat &img);
	void visualize_image_next(cv::Mat &img);
	void record_image(cv::Mat &img);
	void show();
	void eval_mouse(std::string event, int x, int y);
	void eval_key(char key);
	void iterate_to_new_frame();
	void setup_gui();
};

void checkFrameIntegrity(cv::Mat frame);

#endif