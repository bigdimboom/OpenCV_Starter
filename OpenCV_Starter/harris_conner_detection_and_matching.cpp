#include <iostream>                        // std::cout
#include <fstream>                         // std::fstream
#include <cmath>                           // std::abs
#include <opencv2/core/core.hpp>           // cv::Mat
#include <opencv2/highgui/highgui.hpp>     // cv::imread()
#include <string>

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

void ComputeI(Mat& in, Mat& out, Mat& filter)
{
	out = Mat::zeros(in.rows, in.cols, CV_8U);

	for (int r = 1; r < in.rows - 1; ++r)
	{
		for (int c = 1; c < in.cols - 1; ++c)
		{
			double sum = 0.0f;
			for (int rr = -1; rr <= 1; ++rr)
			{
				for (int cc = -1; cc <= 1; ++cc)
				{
					sum = sum + filter.at<uchar>(rr + 1, cc + 1) * in.at<uchar>(r - rr, c - cc);
				}
			}
			out.at<uchar>(r, c) = sum;
		}
	}
}

void applyGaussian(Mat& in, Mat& out, double sigma = 1.0)
{
	// TODO: GenerateFilter
	out = Mat::zeros(in.rows, in.cols, CV_8U);

	Mat G = Mat::zeros(5, 5, CV_64F);
	double left = 2.0 * CV_PI * sigma * sigma;
	// the denominator in the left side of the formula.
	double right = 2.0 * sigma * sigma;
	// the denominator in the left side of the formula.
	double sum = 0.0;
	// sum is for normalization.

	// generate 5x5 kernel
	for (int y = -2; y <= 2; ++y)
	{
		for (int x = -2; x <= 2; ++x)
		{
			double r = sqrt(x*x + y*y);
			G.at<double>(y + 2, x + 2) = (1 / left) * (exp(-(r*r) / right));
			sum += G.at<double>(y + 2, x + 2);
		}
	}

	// normalize the Kernel
	for (int i = 0; i < 5; ++i)
	{
		for (int j = 0; j < 5; ++j)
		{
			G.at<double>(i, j) /= sum;
		}
	}

	// apply to in
	for (int r = 2; r < in.rows - 2; ++r)
	{
		for (int c = 2; c < in.cols - 2; ++c)
		{
			double sum = 0.0f;

			for (int rr = -2; rr <= 2; ++rr)
			{
				for (int cc = -2; cc <= 2; ++cc)
				{
					sum = sum + G.at<double>(rr + 2, cc + 2) * in.at<uchar>(r - rr, c - cc);
				}
			}
			out.at<uchar>(r, c) = sum;
		}
	}
}

int main(int argc, char** argv)
{
	string leftPath = "teddyL.pgm";
	string rightPath = "teddyR.pgm";
	string groundTruthPath = "disp2.pgm";

	//Read Img
	Mat leftImg;
	Mat rightImg;
	Mat groundTruthImg;

	bool loadSucc = false;
	loadSucc = loadImage(leftImg, leftPath);
	loadSucc = loadImage(rightImg, rightPath);
	loadSucc = loadImage(groundTruthImg, groundTruthPath);
	if (!loadSucc)
	{
		return EXIT_FAILURE;
	}

	// STEP 1: compute Is
	Mat leftI_x, leftI_y, leftI_x_smooth, leftI_y_smooth;
	Mat W_x = (Mat_<uchar>(3, 3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
	Mat W_y = (Mat_<uchar>(3, 3) << -1, -1, -1, 0, 0, 0, 1, 1, 1);
	ComputeI(leftImg, leftI_x, W_x);
	ComputeI(leftImg, leftI_y, W_y);

	// STEP 2: Apply Gussian Smooth Filter
	applyGaussian(leftI_x, leftI_x_smooth);
	applyGaussian(leftI_y, leftI_y_smooth);

	//showImage(leftImg, leftPath);
	//showImage(rightImg, rightPath);
	//showImage(groundTruthImg, groundTruthPath);


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
		// succ = imwrite("test.bmp", leftImg);
		if (!succ)
		{
			printf(" Image writing fialed \n ");
			return -1;
		}
	}

	return EXIT_SUCCESS;
}