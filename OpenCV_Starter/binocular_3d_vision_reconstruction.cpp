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

void visualizeDepthMap(Mat& in, Mat& out, int baseLine, int focalLength, double enhanceRate)
{
	out = Mat::zeros(in.rows, in.cols, IMAGE_TYPE_SIZE);

	for (int r = 0; r < out.rows; ++r)
	{
		for (int c = 0; c < out.cols; ++c)
		{
			DepthType depth = in.at<DepthType>(r, c);
			depth = (DepthType)(focalLength * baseLine) / depth;
			depth = depth * enhanceRate;
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
				   int focalLength,
				   bool isDispNegtive = false)
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
			disp = isDispNegtive == true ? -disp : disp;

			if (disp == 0)
			{
				continue; // throw away disparity == 0
			}

			double depthZ = (double)(focalLength * baseLine) / disp; // formula: Z = fb/d.
			double u = (double)c - (double)dispMap.cols / 2.0; // origin is at the center of the image.
			double x = u * depthZ / (double)focalLength; // the x in real world 3D system (x,yz).
			double camX = x + (double)transformX; // x position in the view 3 camera. (from input map postion to view 3).
			int newC = (int)(round(camX * (double)(focalLength) / depthZ
				+ (double)dispMap.cols / 2.0)
				); // project 3d points to uv(2D) coordinate of View 3.
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

	//Mat watch;
	//visualizeDepthMap(out, watch, baseLine, focalLength);
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

// Compute the rank transform in n ¡Á n windows.
bool RankTransform(Mat& input, Mat& out, int windowSize)
{
	if (windowSize % 2 == 0)
	{
		return false; // windows size must be an odd number
	}

	out = Mat::zeros(input.rows, input.cols, IMAGE_TYPE_SIZE);

	int center = (windowSize - 1) / 2;

	for (int r = center; r < out.rows - center; ++r)
	{
		for (int c = center; c < out.cols - center; ++c)
		{
			int rank = 0;

			for (int wr = -center; wr < center; ++wr)
			{
				for (int wc = -center; wc < center; ++wc)
				{
					if (input.at<ImageType>(r + wr, c + wc) < input.at<ImageType>(r, c))
					{
						++rank;
					}
				}
			}

			out.at<ImageType>(r, c) = rank;
		}
	}
	return true;
}

// Compute the SAD in n ¡Á n windows.
bool SAD(Mat& inputLeft, Mat& inputRight, Mat& output, int minDisparity, int maxDisparity, int windowSize)
{
	if (windowSize % 2 == 0)
	{
		return false; // windows size must be an odd number.
	}
	output = Mat::zeros(inputLeft.rows, inputLeft.cols, IMAGE_TYPE_SIZE);
	// the output disperity map.
	int center = (windowSize - 1) / 2;
	for (int r = center; r < inputLeft.rows - center; ++r)
	{
		for (int c = center; c < inputLeft.cols - center; ++c)
		{
			int prevCost = INT_MAX;
			int theMin = minDisparity;

			if (c < center + maxDisparity)
			{
				output.at<ImageType>(r, c) = theMin * 3;
				continue;
			}

			for (int d = 0; d <= maxDisparity; ++d) // slide window
			{
				int currentCost = 0;

				for (int wr = -center; wr < center; ++wr)
				{
					for (int wc = -center; wc < center; ++wc)
					{
						if (c - center - d >= 0)
						{
							int cost = abs(inputLeft.at<ImageType>(r + wr, c + wc) -
										   inputRight.at<ImageType>(r + wr, c + wc - d));
							// difference for one pixel.

							currentCost = currentCost + cost;
							// add all element in the window
							// sum of all pixel differences.
						}
					}
				}
				// Simple ¡°Winner Takes All¡± - Algorithm:
				// For every pixel select the disparity with lowest cost.
				if (prevCost > currentCost)
				{
					prevCost = currentCost;
					theMin = abs(d);
				}
			}
			output.at<ImageType>(r, c) = theMin * 3;
		}
	}
	return true;
}

float computeErrorRate(Mat& sample, Mat& toBeTest, double levelOff = 1.0)
{
	int count = 0;
	for (int r = 0; r < toBeTest.rows && r < sample.rows; ++r)
	{
		for (int c = 0; c < toBeTest.cols && c < sample.cols; ++c)
		{
			double value = sample.at<DepthType>(r, c);
			double testVal = toBeTest.at<DepthType>(r, c);
			if ((testVal < value - levelOff || testVal > value + levelOff))
			{
				++count;
			}
		}
	}

	return  (float(count) / ((float)sample.rows * sample.cols));
}

bool generateReport(string filePath, float errorRate)
{
	ofstream report;
	report.open(filePath);
	if (!report.is_open() || report.fail())
	{
		return false;
	}
	report << "The error is: " << errorRate << endl;
	report.close();
	return true;
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

	string depthMapMixA = "report/depthMap_mix_for_a.bmp";
	string depthMapMixB = "report/depthMap_mix_for_b.bmp";
	string report = "report/report.txt";

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
	Mat depthMap1, depthMap2, theDepthMap, depthImage1;
	generateDepth(depthMap1, img[disp1], -2 * UNIT_BASELINE, UNIT_BASELINE * (5 - 1), FOCAL_LENGTH);
	generateDepth(depthMap2, img[disp5], 2 * UNIT_BASELINE, -UNIT_BASELINE * (5 - 1), FOCAL_LENGTH, true);

	// STEP 2: project two depth maps to view3, pick the value that is closer to camera.
	combine(theDepthMap, depthMap1, depthMap2);

	// STEP 3: visualize the depth map
	visualizeDepthMap(theDepthMap, depthImage1, UNIT_BASELINE * (5 - 1), FOCAL_LENGTH, 3.0);
	showImage(depthImage1, depthMapMixA);

	// TODO: main steps for (b)
	// STEP 1: calculate 6 rank maps.
	vector<Mat> rankMaps;
	bool extractSucc = false;
	for (auto i = img.begin(); i != img.end(); ++i)
	{
		if (i->first != disp1 && i->first != disp5)
		{
			Mat tmp;
			extractSucc = RankTransform(i->second, tmp, 5);
			if (!extractSucc)
			{
				std::cerr << "rank compute filed\n";
				return EXIT_FAILURE;
			}
			rankMaps.push_back(tmp);
		}
	}

	// STEP 2: calculate 3 disparity maps.
	vector<Mat> dispMaps;
	for (int i = 1; i < rankMaps.size(); i = i + 2)
	{
		Mat tmp;
		extractSucc = SAD(rankMaps[i], rankMaps[i + 1], tmp, 0, 22, 9);
		// Why 0 - 22? Because disparity between view 1 - 5 is 0 - 85.
		if (!extractSucc)
		{
			std::cerr << "disparity compute filed\n";
			return EXIT_FAILURE;
		}
		dispMaps.push_back(tmp);
	}
	//for (int i = 0; i < dispMaps.size(); ++i)
	//{
	//	showImage(dispMaps[i], to_string(i));
	//}

	// STEP 3: Generate 3 depthMap for peoblem 2 and combine them to one.
	Mat depth1, depth2, depth3, theDepth, depthImage2;
	generateDepth(depth1, dispMaps[0], -2 * UNIT_BASELINE, UNIT_BASELINE, FOCAL_LENGTH);
	generateDepth(depth2, dispMaps[1], 0, UNIT_BASELINE, FOCAL_LENGTH);
	generateDepth(depth3, dispMaps[2], 2 * UNIT_BASELINE, UNIT_BASELINE, FOCAL_LENGTH);
	Mat tmpDepthMap;
	combine(tmpDepthMap, depth1, depth2);
	combine(theDepth, tmpDepthMap, depth3);

	// STEP 4: visualize the depthMap for problem 3.
	visualizeDepthMap(theDepth, depthImage2, UNIT_BASELINE, FOCAL_LENGTH, 85.0 / 22.0 * 3.0);
	showImage(depthImage2, depthMapMixB);

	// STEP 5: calculate error rate theDepth :: theDepthMap
	float errorRate = computeErrorRate(theDepthMap, theDepth);
	//float errorRate = computeErrorRate(depthImage1, depthImage2);
	cout << "Error rate is: " << errorRate << endl;


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
		succ = imwrite(depthMapMixA, depthImage1);
		succ = imwrite(depthMapMixB, depthImage2);
		succ = generateReport(report, errorRate);
		if (!succ)
		{
			printf(" Image writing fialed \n ");
			return -1;
		}
	}

	return EXIT_SUCCESS;
}