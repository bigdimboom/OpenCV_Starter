#include <iostream>                        // std::cout
#include <fstream>                         // std::fstream
#include <cmath>                           // std::abs
#include <string>                          // std::string
#include <vector>                          // std::vector
#include <map>                             // std::map
#include <opencv2/core/core.hpp>           // cv::Mat
#include <opencv2/highgui/highgui.hpp>     // cv::imread()
#include <opencv2/imgproc/imgproc.hpp>     // image process lib opencv
#include <assert.h>						   // assert(...)


#define FOCAL_LENGTH 1247 // pixels
#define UNIT_BASELINE 40  // millimeters

#define DEPTH_TYPE_SIZE CV_64F
#define DEPTH_INFINITY 999999
typedef double DepthType;

#define IMAGE_TYPE_SIZE CV_8U
#define MAX_COLOR 255
#define MIN_COLOR 0
typedef uchar ImageType;

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

int loadMany(Mat* img, std::string* path, int size)
{
	bool ret = false;
	for (int i = 0; i < size; ++i)
	{
		ret = loadImage(img[i], path[i]);
		if (!ret)
		{
			return i;
		}
		img[i].convertTo(img[i], DEPTH_TYPE_SIZE);
	}
	return -1;
}

void showMany(Mat* img, std::string* path, int size)
{
	for (int i = 0; i < size; ++i)
	{
		showImage(img[i], path[i]);
	}
}

bool loadCamMat(Mat* mat, int size, const string & path)
{
	std::fstream myfile(path, std::ios_base::in);
	vector<double> d_test;
	if (myfile.fail() || !myfile.is_open())
	{
		cerr << "file opening failed\n";
		return false;
	}
	double a;
	while (myfile >> a)
	{
		d_test.push_back(a);
	}
	myfile.close();

	for (int i = 0; i < size; ++i)
	{
		mat[i] = Mat::zeros(3, 4, DEPTH_TYPE_SIZE);
	}

	int count = 0;
	for (int i = 0; i < size; ++i)
	{
		for (int r = 0; r < 3; ++r)
		{
			for (int c = 0; c < 4; ++c)
			{
				mat[i].at<DepthType>(r, c) = d_test[count];
				++count;
			}
		}
	}
	return true;
}

int main(int argc, char** argv)
{
	Mat silhImg[8];
	string silhPath[8] = {
		"silh/silh_cam00_00023_0000008550.pbm",
		"silh/silh_cam01_00023_0000008550.pbm",
		"silh/silh_cam02_00023_0000008550.pbm",
		"silh/silh_cam03_00023_0000008550.pbm",
		"silh/silh_cam04_00023_0000008550.pbm",
		"silh/silh_cam05_00023_0000008550.pbm",
		"silh/silh_cam06_00023_0000008550.pbm",
		"silh/silh_cam07_00023_0000008550.pbm"
	};
	int errorPictNum;
	if (errorPictNum = loadMany(silhImg, silhPath, 8) != -1)
	{
		std::cerr << "Loading Silh files fiailed : (" << errorPictNum << ").\n";
		return EXIT_FAILURE;
	}
	// showMany(silhImg, silhPath, 8);

	Mat camMat[8];
	loadCamMat(camMat, 8, "cameras.txt");

	//for (int i = 0; i < 8; ++i)
	//{
	//	std::cout << camMat[i] <<endl;
	//}

	//TODO:
	// STEP 1: createing voxels
	const double res = 0.001; // (1/1000)
	Point2d rangeX(-2.5, 2.5), rangeY(-3.0,3.0), rangeZ(0.0,2.5);
	double sizeX, sizeY, sizeZ;
	sizeX = (rangeX.y - rangeX.x) * res;
	sizeY = (rangeY.y - rangeY.x) * res;
	sizeZ = (rangeZ.y - rangeZ.x) * res;




	std::cout << "Press \"s\" to save, \"esc\" to close the program.\n";
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

		if (!succ)
		{
			printf(" Image writing fialed \n ");
			return -1;
		}
	}

	return EXIT_SUCCESS;
}