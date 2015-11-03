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
#define DEPTH_INFINITY 9999999
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

void visualizeDepthMap(Mat& in, Mat& out, int baseLine, int focalLength)
{
	out = Mat::zeros(in.rows, in.cols, IMAGE_TYPE_SIZE);

	for (int r = 0; r < out.rows; ++r)
	{
		for (int c = 0; c < out.cols; ++c)
		{
			DepthType depth = in.at<DepthType>(r, c);
			depth = (DepthType)(focalLength * baseLine) / depth;
			depth = depth * 3;
			depth = depth < MIN_COLOR ? MIN_COLOR : depth;
			depth = depth > MAX_COLOR ? MAX_COLOR : depth;
			out.at<ImageType>(r, c) = (ImageType)depth;
		}
	}

}

// pretend disMap is the world center;
// then make "out" the world center;
// then project 3d points to the "out" view 3  image 2D system.
void generateDepth(Mat& out, // view 3
				   const Mat& dispMap,
				   int transformX, // (0,0,0) is the "out" map position; dispMap position should be compare to it.
				   int baseLine,
				   int focalLength)
{
	out = Mat::zeros(dispMap.rows, dispMap.cols, DEPTH_TYPE_SIZE);

	for (int r = 0; r < out.rows; ++r)
	{
		for (int c = 0; c < out.cols; ++c)
		{
			out.at<DepthType>(r, c) = DEPTH_INFINITY;
		}
	}

	for (int r = 0; r < dispMap.rows; ++r)
	{
		for (int c = 0; c < dispMap.cols; ++c)
		{
			double disp = (double)dispMap.at<uchar>(r, c) / 3.0;

			if (disp == 0)
			{
				continue; // throw away disparity == 0
			}

			double depthZ = (double)(focalLength * baseLine) / disp; // formula: Z = fb/d.
			double u = (double)c - (double)dispMap.cols/ 2.0; // origin is at the center of the image.
			double x = u * depthZ / (double)focalLength; // the x in real world 3D system (x,yz).
			double camX = x + (double)transformX; // x position in the view 3 camera. (from input map postion to view 3).
			int newC = (int) (round(camX * (double)(focalLength) / depthZ 
								+ (double)dispMap.cols / 2.0)
							 ); // project 3d points to 2D image system of View 3.
			// move horizontally; therefore, y values do not change.

			if (newC >= 0 && newC < out.cols)
			{
				if (depthZ < out.at<DepthType>(r, newC))
				{
					out.at<DepthType>(r, newC) = depthZ;
				}
			}
		}
	}
}

void combine(Mat& out, const Mat& depthMap1, const Mat& depthMap2)
{
	assert(depthMap1.cols == depthMap2.cols
		   && depthMap1.rows == depthMap2.rows);

	out = Mat::zeros(depthMap1.rows, depthMap1.cols, DEPTH_TYPE_SIZE);

	for (int r = 0; r < out.rows; ++r)
	{
		for (int c = 0; c < out.cols; ++c)
		{
			DepthType one = depthMap1.at<DepthType>(r, c);
			DepthType two = depthMap2.at<DepthType>(r, c);
			out.at<DepthType>(r, c) = one < two ? one : two;
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

	// TODO: main steps for (a)
	// STEP 1: Generate depth for view 3 from ground truth view 1 and view 5
	Mat depthMap1, depthMap2, theDepth, depthImage;
	generateDepth(depthMap1, img[disp1], -2 * UNIT_BASELINE, UNIT_BASELINE * (5 - 1), FOCAL_LENGTH);
	generateDepth(depthMap2, img[disp5],  2 * UNIT_BASELINE, UNIT_BASELINE * (5 - 1), FOCAL_LENGTH);
	// STEP 2: project two depth maps to view3, pick the value that is closer to camera.
	combine(theDepth, depthMap1, depthMap2);
	visualizeDepthMap(theDepth, depthImage, UNIT_BASELINE * (5 - 1), FOCAL_LENGTH);

	showImage(depthImage, depthMapMix);

	// TODO: main steps for (b)
	// STEP 1: calculate 3 disperity maps.



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
		succ = imwrite(depthMapMix, depthImage);
		if (!succ)
		{
			printf(" Image writing fialed \n ");
			return -1;
		}
	}

	return EXIT_SUCCESS;
}