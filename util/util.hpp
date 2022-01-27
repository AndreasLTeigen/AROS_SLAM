#ifndef util_h
#define util_h

std::string zeroPad(int num, int pad_n);

void reduceImgContrast(cv::Mat img, int lower_level=128, int upper_level=255);
void drawCircle(cv::Mat &img, cv::Point point, int radius=15);
void drawCircle(cv::Mat &img, cv::Mat point, int radius=15);
cv::Mat fundamentalFromEssential(cv::Mat E_matrix, cv::Mat K_matrix);
cv::Mat calculateEpipole(cv::Mat E_matrix);
void drawEpipolarLines(cv::Mat F, cv::Mat &img_disp2,
                        std::vector<cv::Point> points1,
                        std::vector<cv::Point> points2);

cv::Mat compileKMatrix( double fx, double fy, double cx, double cy );

bool isRotationMatrix(cv::Mat &R);
std::vector<double>  rotationMatrixToEulerAngles(cv::Mat &R);

cv::Mat T2Rot(cv::Mat &T);

cv::Mat T2Trans(cv::Mat &T);

void T2RotAndTrans(cv::Mat &T, cv::Mat &R, cv::Mat &t);

cv::Mat inverseTMatrix(cv::Mat T);

cv::Mat compileTMatrix(std::vector<double> pose);

cv::Mat xy2Mat(double x, double y);

cv::Mat xyToxy1(double x, double y);

void xyToxy1(cv::Mat& xy);

void dehomogenizeMatrix(cv::Mat& X);

cv::Mat dilateKptWDepth(cv::Mat xy1, double Z, cv::Mat T, cv::Mat K);

cv::Mat projectKpt( cv::Mat XYZ, cv::Mat T, cv::Mat K );

cv::Mat relTfromglobalTx2(cv::Mat T1, cv::Mat T2);

std::vector<double> transform2stdParam(cv::Mat &T);

void writeParameters2File(std::string file_path, std::string image_idenifier, cv::Mat &T );

std::vector<std::vector<std::string>> readCSVFile(std::string filename);

#endif