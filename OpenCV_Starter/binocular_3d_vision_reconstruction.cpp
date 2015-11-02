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

#define TYPE_SIZE CV_8U
#define MAX_COLOR UCHAR_MAX
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

// pretend disMap is the world center;
// then make "out" the world center;
// then transfer 3d points to the "out" system.
void generateDepth(Mat& out,
				   const Mat& dispMap,
				   double dispMapPosX, // (0,0,0) is the "out" map position; dispMap position should be compare to it.
				   int baseLine,
				   int focalLength)
{
	out = Mat::zeros(dispMap.rows, dispMap.cols, TYPE_SIZE);

	for (int r = 0; r < out.rows; ++r)
	{
		for (int c = 0; c < out.cols; ++c)
		{
			out.at<ImageType>(r, c) = MAX_COLOR;
		}
	}

	for (int r = 0; r < dispMap.rows; ++r)
	{
		for (int c = 0; c < dispMap.cols; ++c)
		{
			double disp = dispMap.at<uchar>(r, c) / 3.0;
			if (disp == 0)
			{
				continue;
			}
			double depthZ = ((double)focalLength * (double)baseLine / disp);
			double u = (double)c - (double)dispMap.cols / 2.0; // u is the x coodinate for the disparity map.
			double x = (u * depthZ / focalLength) + dispMapPosX; // x is the x coodinate in view 3 camera system.
			// y in 3D will not change.
			double uu = x * focalLength / depthZ; // the x of view 3 in 2D
			int newC = (int)(round(uu) + (double)dispMap.cols / 2.0);

			if (newC > 0 && newC < out.cols)
			{
				out.at<ImageType>(r, newC) = (ImageType)(round((double)focalLength * (double)baseLine / depthZ) * 3.0);
			}
		}
	}
}

void combine(Mat& out, const Mat& depthMap1, const Mat& depthMap2)
{
	assert(depthMap1.cols == depthMap2.cols
		   && depthMap1.rows == depthMap2.rows);

	out = Mat::zeros(depthMap1.rows, depthMap1.cols, TYPE_SIZE);

	for (int r = 0; r < out.rows; ++r)
	{
		for (int c = 0; c < out.cols; ++c)
		{
			ImageType one = depthMap1.at<ImageType>(r, c);
			ImageType two = depthMap2.at<ImageType>(r, c);
			out.at<ImageType>(r, c) = one <= two ? one : two;
		}
	}
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

	string depthMapMix = "report/depthMap_mix.bmp";

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
	// STEP 1: Generate depth for view 3 from ground truth view 1 and view 5
	Mat depthMap1, depthMap2, theDepth;
	generateDepth(depthMap1, img[disp1], -(UNIT_BASELINE * 2), UNIT_BASELINE, FOCAL_LENGTH);
	generateDepth(depthMap2, img[disp5], UNIT_BASELINE * 2, UNIT_BASELINE, FOCAL_LENGTH);
	combine(theDepth, depthMap1, depthMap2);

	Mat test1 = img[disp1];
	Mat test2 = img[disp5];
	showImage(theDepth, depthMapMix);

	//for (auto i = img.begin(); i != img.end(); ++i)
	//{
	//	showImage(i->second, i->first);
	//}

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
		succ = imwrite(depthMapMix, theDepth);
		if (!succ)
		{
			printf(" Image writing fialed \n ");
			return -1;
		}
	}

	return EXIT_SUCCESS;
}