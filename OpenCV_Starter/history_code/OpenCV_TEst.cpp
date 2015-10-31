#include <iostream>                        // std::cout
#include <opencv2/core/core.hpp>           // cv::Mat
#include <opencv2/highgui/highgui.hpp>     // cv::imread()
#include <opencv2/imgproc/imgproc.hpp>     // cv::getPerspective()

using namespace std;
using namespace cv;

int main()
{
	Point3f pts(1, 2, 3);

	Mat C = (Mat_<float>(3, 3) << 1,0,2, 0, 1, 3, 0, 0, 1);

	Mat ret = C * Mat(pts);

	Point3f retPtr(ret);

	cout << ret << endl;
	cout << retPtr << endl;

	return 0;
}