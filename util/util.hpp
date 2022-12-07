#ifndef util_h
#define util_h

std::string zeroPad(int num, int pad_n);

double iterativeAverage(double old_mean, double new_val, int n);

void reduceImgContrast(cv::Mat img, int lower_level=128, int upper_level=255);

void drawCircle(cv::Mat &img, cv::Point point, int radius=15);

void drawCircle(cv::Mat &img, cv::Mat point, int radius=15);

void drawIndicator(cv::Mat& img, double percentage, cv::Point pos);

cv::Scalar percentage2Color(double p);

cv::Mat invertKMatrix( cv::Mat K );

cv::Mat composeEMatrix(cv::Mat& R, cv::Mat& t);

cv::Mat fundamentalFromEssential(cv::Mat E_matrix, cv::Mat K_matrix);

cv::Mat fundamentalFromEssential(cv::Mat E_matrix, cv::Mat K1_matrix, cv::Mat K2_matrix);

cv::Mat calculateEpipole(cv::Mat E_matrix);

void drawEpipolarLines(cv::Mat F, cv::Mat &img_disp2,
                        std::vector<cv::Point2f> points1,
                        std::vector<cv::Point2f> points2);

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

void homogenizeArray(cv::Mat& xy);

cv::Mat homogenizeArrayRet(const cv::Mat& arr);

void dehomogenizeMatrix(cv::Mat& X);

cv::Mat normalizeMat(cv::Mat& vec);

cv::Mat fitQuadraticForm(cv::Mat& x, cv::Mat& y, cv::Mat& z);

cv::Mat sampleQuadraticForm(cv::Mat A, cv::Point center, cv::Size reg_size );

cv::Mat reprojectionError( cv::Mat& xyz1, cv::Mat& uv1, cv::Mat& T, cv::Mat& K );

cv::Mat dilateKptWDepth(cv::Mat xy1, double Z, cv::Mat T, cv::Mat K);

cv::Mat projectKpt( cv::Mat XYZ1, cv::Mat T, cv::Mat K );

void triangulatePointsLinear( cv::Mat& rel_T, cv::Mat& K1, cv::Mat& K2, cv::Mat& uv1, cv::Mat& uv2, cv::Mat& XYZ_I2 );

cv::Mat computeHammingDistance( cv::Mat& target_desc, cv::Mat& region_descs );

cv::Mat relTfromglobalTx2(cv::Mat T1, cv::Mat T2);

std::vector<double> transform2stdParam(cv::Mat &T);

bool isInsideImageBounds( double y, double x, int H, int W);

static void meshgrid( const cv::Mat& xgv, const cv::Mat& ygv, cv::Mat& X, cv::Mat& Y );

std::string type2str(int type);

void writeParameters2File(std::string file_path, std::string image_idenifier, cv::Mat &T );

void writeTransformation2File(std::string file_path, std::string image_idenifier, cv::Mat &T );

void writeVector2File(std::string file_path, std::vector<double> &data, bool linebreak=true);

void clearFile(std::string file_path);

std::vector<std::vector<std::string>> readCSVFile(std::string filename, char delim);

bool saveImage( cv::Mat& img, std::string img_name, std::string folder );

#endif