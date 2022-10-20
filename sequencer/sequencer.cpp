#include <iostream>
#include <opencv2/opencv.hpp>

#include "sequencer.hpp"

using std::string;

void checkFrameIntegrity(cv::Mat frame)
{
    if(frame.empty())
    {
        throw "Error: Frame could not be loaded! \nTerminating program";
    }
}

#include <stdio.h>
#ifdef __linux__
	#include <dirent.h>
	#include <sys/stat.h>
	#include <sys/types.h>
#elif (_WIN32|| _WIN64)
	#include <Windows.h>
#endif
using namespace std;
using namespace cv;

Sequencer::Sequencer(std::string folder_path, int frame_buffer_size, string file_format, bool recording, string record_dst_path, int fps, bool auto_start){
	this->frame_buffer_size = frame_buffer_size;
	this->pressed_key = -1;
	this->fps = fps;
	this->file_format = file_format;
	this->play_speed = 1;
	this->slow_motion = false;
	this->just_finished = false;
	this->finished = false;
	this->recording = recording;
	this->record_dst_path = record_dst_path;

	if (auto_start)
	{
		this->play_mode = true;
	}
	else
	{
		this->play_mode = false;
	}
	
	// @TODO: add visualization parameters
	this->scale = 1;

	if (folder_path.back() != '/' && folder_path.back() != '\\'){
		this->path = folder_path + "/";
	}
	else{
		this->path = folder_path;
	}
#ifdef __linux__ 
	DIR *dp;
	struct dirent *dirp;
	if ((dp = opendir(folder_path.c_str())) == NULL) {
		cout << "Error(" << errno << ") opening " << folder_path << endl;
	}
	string filepath_temp;
	struct stat filestat;
	while ((dirp = readdir(dp)) != NULL) {
		 filepath_temp = folder_path + "/" + dirp->d_name;
		// If the file is a directory (or is in some way invalid) we'll skip it
		if (stat( filepath_temp.c_str(), &filestat )) continue;
		if (S_ISDIR( filestat.st_mode ))         continue;
		img_seq.push_back(string(dirp->d_name));
	}
	closedir(dp);

#elif (_WIN32 || _WIN64)
	WIN32_FIND_DATA fd;
	HANDLE h = FindFirstFile((this->path + "*." + this->file_format).c_str(), &fd); // 
	while (1){
		img_seq.push_back(fd.cFileName);
		if (FindNextFile(h, &fd) == FALSE)
			break;
	}
#endif
	sort(img_seq.begin(), img_seq.end());
	current_index = 0;
	max_index = img_seq.size() - 1;
	if (max_index >= 0){
		setupGui();
	}
	else{
		cout << "No images found in folder " << folder_path << endl;
	}
}


Sequencer::~Sequencer(){
	// @TODO: create destructor
}

int Sequencer::getFrameBufferSize()
{
	return this->frame_buffer_size;
}

bool Sequencer::hasNextFrame(){
	return (this->current_index <= this->max_index);
}


Mat Sequencer::getCurrentFrame(){
	string img_path = this->path + this->img_seq[this->current_index];
	Mat imgCurrent = imread(img_path, cv::IMREAD_GRAYSCALE);

	this->width = imgCurrent.cols;
	this->height = imgCurrent.rows;
	return imgCurrent;
}

Mat Sequencer::getNextFrame(){
	string img_path = this->path + this->img_seq[this->current_index + this->play_speed];
	Mat imgCurrent = imread(img_path, cv::IMREAD_GRAYSCALE);
	return imgCurrent;
}

string Sequencer::getCurrentPath(){
	return this->path + this->img_seq[this->current_index];
}

string Sequencer::getCurrentName()
{
	return this->img_seq[this->current_index];
}

int Sequencer::getCurrentIndex(){
	return this->current_index;
}

void Sequencer::setCurrentIndex(int index){
	this->current_index = index;
}

Point2d Sequencer::getMouseCoords(){
	return this->mouse_coords;
}

int Sequencer::getPressedKey(){
	return this->pressed_key;
}

bool Sequencer::isJustFinished(){
	return this->just_finished;
}

bool Sequencer::isFinished(){
	return this->finished;
}

void Sequencer::preLoadFrameBufferFull()
{

	/* Pre-loading frame FIFO buffer of size N and sets current index 
	   to n+1. */

	Mat frame;
	bool tmp = this->play_mode;

	this->play_mode = true;

	cout << "Loading frame buffer: " << endl;
	
	for (int i = 0; i < this->frame_buffer_size; i++)
	{
		frame = this->getCurrentFrame();
		this->frame_buffer.push (frame);
		cout << "Pre-loading frame nr: " << this->getCurrentIndex() << endl;
		this->iterateToNewFrame();
	}
	this->play_mode = tmp;
}

void Sequencer::preLoadFrameBufferFrame(Mat frame)
{
	/* Loading frame into frame buffer and iterating the sequencer */

	bool tmp = this->play_mode;

	cout << "Pre-loading frame nr: " << this->getCurrentIndex() << endl;
	this->play_mode = true;
	this->frame_buffer.push( frame );
	this->iterateToNewFrame();
	this->play_mode = tmp;

}

void Sequencer::frameBufferPush(Mat frame)
{
	this->frame_buffer.push(frame);
	this->frame_buffer.pop();
}

Mat Sequencer::frameBufferGetNewest()
{
	return this->frame_buffer.back();
}

Mat Sequencer::frameBufferGetOldest()
{
	return this->frame_buffer.front();
}

void Sequencer::visualizeImage(Mat &img){

	/* Visualizes the image and also links to hotkey based UI */

	this->img_current = img;
	this->mouse_coords = Point2d(-1, -1);
	if (img.channels() == 1){
		cvtColor(this->img_current, this->out, COLOR_GRAY2BGR);
	}
	else{
		this->out = this->img_current;
	}

	this->show();

	if (recording)
	{
		this->recordImage(this->img_current);
	}

	if (this->play_mode && this->slow_motion){
		this->evalKey(waitKey(4 * 1000 / this->fps));
	}
	else if (this->play_mode){
		//this->evalKey(waitKey(1000 / this->fps));
		this->evalKey(waitKey(1));
	}
	else{
		this->evalKey(waitKey(0));
	}
}


void Sequencer::visualizeImageNext(Mat &img){
	Mat img_resized;
	resize(img, img_resized, Size(this->scale, this->scale));
	imshow(this->path, img_resized);
	//moveWindow("img_current", this->origin_offset.x, this->origin_offset.y + img_resized.cols + 120);
}

void Sequencer::recordImage(Mat &img)
{
	std::string record_path = this->record_dst_path + this->img_seq[this->current_index];
	bool check = imwrite(record_path, img);
	if (check == false) 
	{
        cout << "Mission - Saving the image, FAILED" << endl;
	}
}

void Sequencer::show(){
	string mode;
	string speed;
	if (this->play_mode){
		mode = "I>";
	}
	else{
		mode = "II";
	}

	if (this->slow_motion){
		speed = "slow";
	}
	else{
		speed = std::to_string(this->play_speed);
		speed = speed + "x";
	}

	float percent = 100 * float(this->current_index) / float(this->max_index);
	char buff[10];
#ifdef __linux__
	snprintf(buff, 10, "%05.01f%%", percent);
#elif (_WIN32 || _WIN64)
	sprintf_s(buff, "%05.01f%%", percent);
#endif
	std::string percent_txt = buff;
	putText(this->out, mode, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
	putText(this->out, speed, Point(50, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
	putText(this->out, percent_txt, Point(this->width - 120, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);

	Mat img_concat;
	hconcat(this->out, this->control_panel, img_concat);
	// TODO add resize
	//resize(img_concat, this->out, Size(this->scale, this->scale));
	imshow(this->path, img_concat);
	moveWindow(this->path, int(this->origin_offset.x), int(this->origin_offset.y));
	// TODO: Set mouse call back
	//setMouseCallback
}

//void Sequencer::evalMouse(string event, int x, int y);

void Sequencer::evalKey(char key){

	/* Handles the hotkeybased UI given a keypress input */

	key = int(key & 255);
	if (key == 'd'){
		this->current_index = min(this->max_index, this->current_index + this->play_speed);
	}
	else if (key == 'a'){
		this->current_index = max(0, this->current_index - 1);
	}
	else if (key == 'w'){
		this->current_index = min(this->max_index, this->current_index + this->fps);
	}
	else if (key == 's'){
		this->current_index = max(0, this->current_index - this->fps);
	}
	else if (key == 'q' || key == 27){
		this->play_mode = false;
		this->current_index = this->max_index + 1;
		cv::destroyAllWindows();
	}
	else if (key == 'r'){
		this->play_mode = !this->play_mode;
	}
	else if (key == 'e'){
		this->slow_motion = !this->slow_motion;
	}
	else if (key >= '1' && key <= '9'){
		this->play_speed = key - '0';
		cout << "new speed =" << key - '0' << endl;
	}
	else if (key == 'y'){
		this->current_index = 0;
	}
	else if (key == 'x'){
		this->current_index = int(this->max_index / 2);
	}
	else if (key == 'c'){
		this->current_index = this->max_index;
	}
}
void Sequencer::iterateToNewFrame(){

	/* Given different methods of playback, shift current index to the
	 new frame */

	if (this->play_mode & !this->slow_motion){
		this->current_index = min(this->max_index, this->current_index + this->play_speed);
	}
	else if (this->play_mode){
		this->current_index = min(this->max_index, this->current_index + 1);
	}
	if (!this->finished && (this->current_index == (this->max_index - 1))){
		this->finished = true;
		this->just_finished = true;
	}
	else{
		this->just_finished = false;
	}
}

void Sequencer::setupGui(){
	string img_path = this->path + this->img_seq[this->current_index];
	Mat img = imread(img_path, cv::IMREAD_GRAYSCALE);
	this->width = img.cols;
	this->height = img.rows;

	this->control_panel = Mat::zeros(Size(500, this->height), CV_8UC3);

	putText(this->control_panel, "Control Keys", Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
	putText(this->control_panel, "w     : forward 1s", Point(10, 30 * 3), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
	putText(this->control_panel, "s     : backward 1s", Point(10, 30 * 4), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
	putText(this->control_panel, "a     : previous image", Point(10, 30 * 5), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
	putText(this->control_panel, "d     : next image", Point(10, 30 * 6), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
	putText(this->control_panel, "r     : play/pause", Point(10, 30 * 7), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
	putText(this->control_panel, "e     : slow/normal", Point(10, 30 * 8), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
	putText(this->control_panel, "1-9   : 1x - 9x speed", Point(10, 30 * 9), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
	putText(this->control_panel, "q,esc : quit", Point(10, 30 * 10), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
	putText(this->control_panel, "y,x,c : beginning/middle/end", Point(10, 30 * 11), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
};
