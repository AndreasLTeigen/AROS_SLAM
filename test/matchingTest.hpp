#ifndef matchingTest_h
#define matchingTest_h

void matchingTest();
void doORBDetectAndCompute(cv::Mat& frame, std::vector<cv::KeyPoint>& kpts, cv::Mat& desc);
void doBruteForceMatchingMono( cv::Mat& queryDesc, cv::Mat& trainDesc, std::vector<std::vector<cv::DMatch>>& matches );
void drawMatchTrails(cv::Mat &img, std::vector<cv::Point> pts1, std::vector<cv::Point> pts2);

#endif