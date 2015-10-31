#include <iostream>                        // std::cout
#include <fstream>                         // std::fstream
#include <cmath>                           // std::abs
#include <string>                          // std::string
#include <vector>                          // std::vector
#include <map>                             // std::map
#include <opencv2/core/core.hpp>           // cv::Mat
#include <opencv2/highgui/highgui.hpp>     // cv::imread()
#include <opencv2/imgproc/imgproc.hpp>     // image process lib opencv

using namespace std;
using namespace cv;

bool loadImage(Mat& img, string path)
{
	img = imread(path, -1);
	if (!img.data)
	{
		std::cerr << "No such data: " << path << "\n.";
		return false;
	}
	return true;
}

void showImage(Mat& img, string path)
{
	namedWindow(path, CV_WINDOW_AUTOSIZE);
	imshow(path, img);
}

int main(int argc, char** argv)
{
	string disp1 = "images/disp1.pgm";
	string disp5 = "images/disp5.pgm";
	string view0 = "images/view0.pgm";
	string view1 = "images/view1.pgm";
	string view2 = "images/view2.pgm";
	string view3 = "images/view3.pgm";
	string view4 = "images/view4.pgm";
	string view5 = "images/view5.pgm";
	string view6 = "images/view6.pgm";

	//Read Img
	map<string, Mat> img;

	bool loadSucc = false;
	loadSucc = loadImage(img[disp1], disp1);
	loadSucc = loadImage(img[disp5], disp5);
	loadSucc = loadImage(img[view0], view0);
	loadSucc = loadImage(img[view1], view1);
	loadSucc = loadImage(img[view2], view2);
	loadSucc = loadImage(img[view3], view3);
	loadSucc = loadImage(img[view4], view4);
	loadSucc = loadImage(img[view5], view5);
	loadSucc = loadImage(img[view6], view6);

	if (!loadSucc)
	{
		return EXIT_FAILURE;
	}

	// TODO: main steps



	for (auto i = img.begin(); i != img.end(); ++i)
	{
		showImage(i->second, i->first);
	}
	cout << "Press \"s\" to save, \"esc\" to close the program.\n";
	int key = waitKey(0);

	// Check if result is correct
	if (key == 27)
	{
		destroyAllWindows();
	}
	else if (key == 's' || key == 'S')
	{
		// Write Output
		bool succ = false;
		// succ = imwrite("report/test.bmp", view0Img);
		if (!succ)
		{
			printf(" Image writing fialed \n ");
			return -1;
		}
	}

	return EXIT_SUCCESS;
}