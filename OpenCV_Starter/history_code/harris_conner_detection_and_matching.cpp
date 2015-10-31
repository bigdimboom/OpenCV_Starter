#include <iostream>                        // std::cout
#include <fstream>                         // std::fstream
#include <cmath>                           // std::abs
#include <opencv2/core/core.hpp>           // cv::Mat
#include <opencv2/highgui/highgui.hpp>     // cv::imread()
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <vector>

using namespace std;
using namespace cv;

#define TYPE_SIZE CV_64F
typedef double ImageType;

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
	out = Mat::zeros(in.rows, in.cols, TYPE_SIZE);

	for (int r = 1; r < in.rows - 1; ++r)
	{
		for (int c = 1; c < in.cols - 1; ++c)
		{
			double sum = 0.0f;
			for (int rr = -1; rr <= 1; ++rr)
			{
				for (int cc = -1; cc <= 1; ++cc)
				{
					sum = sum + filter.at<int>(rr + 1, cc + 1) * in.at<uchar>(r - rr, c - cc);
				}
			}
			sum = sum < 0 ? 0 : sum;
			sum = sum > 255 ? 255 : sum;
			out.at<ImageType>(r, c) = sum;
		}
	}
}

void applyGaussian(Mat& in, Mat& out, double sigma = 1.0)
{
	out = Mat::zeros(in.rows, in.cols, TYPE_SIZE);
	Mat G = Mat::zeros(5, 5, CV_64F);
	// init Gussian smooth filter
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

	// apply G to in
	for (int r = 2; r < in.rows - 2; ++r)
	{
		for (int c = 2; c < in.cols - 2; ++c)
		{
			double sum = 0.0f;

			for (int rr = -2; rr <= 2; ++rr)
			{
				for (int cc = -2; cc <= 2; ++cc)
				{
					sum = sum + G.at<double>(rr + 2, cc + 2) * in.at<ImageType>(r - rr, c - cc);
				}
			}
			out.at<ImageType>(r, c) = sum;
		}
	}
}

void harrisOperation(Mat& Ixx, Mat& Iyy, Mat& Ixy, Mat& out, double threshold = 2000)
{
	out = Mat::zeros(Ixx.rows, Ixx.cols, TYPE_SIZE);
	// int count = 0;

	for (int r = 2; r < out.rows; ++r)
	{
		for (int c = 2; c < out.cols; ++c)
		{
			double f = 0.0;
			double xx = Ixx.at<ImageType>(r, c);
			double yy = Iyy.at<ImageType>(r, c);
			double xy = Ixy.at<ImageType>(r, c);

			double detOfMat = xx * yy - xy * xy;
			double traceOfMat = xx + yy;

			f = detOfMat / traceOfMat;
			// slides: set 4, page 61.

			if (f > threshold)
			{
				out.at<ImageType>(r, c) = f;
				// ++count.
			}
		}
	}

	// cout << count.
}

void applyNonMaxSuppression(Mat& in)
{
	for (int r = 1; r < in.rows - 1; ++r)
	{
		for (int c = 1; c < in.cols - 1; ++c)
		{
			double sum = in.at<ImageType>(r, c);

			bool leave = false;

			for (int rr = -1; rr <= 1 && leave == false; ++rr)
			{
				for (int cc = -1; cc <= 1 && leave == false; ++cc)
				{
					if (in.at<ImageType>(r - rr, c - cc) > sum)
					{
						in.at<ImageType>(r, c) = 0;
						leave = true;
					}
				}
			}

		}
	}
}

void execFeatureExtraction(Mat& in, Mat&out)
{
	// STEP 1: compute Is
	Mat Ix, Iy, Ixx, Iyy, Ixy;
	Mat W_x = (Mat_<int>(3, 3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
	Mat W_y = (Mat_<int>(3, 3) << -1, -1, -1, 0, 0, 0, 1, 1, 1);

	ComputeI(in, Ix, W_x);
	ComputeI(in, Iy, W_y);

	Ixx = Ix.mul(Ix); // I(x*x)
	Iyy = Iy.mul(Iy); // I(y*y)
	Ixy = Ix.mul(Iy); // I(x*y)

	// STEP 2: Apply Gussian Smooth Filter
	Mat Ixx_smooth, Iyy_smooth, Ixy_smooth;
	applyGaussian(Ixx, Ixx_smooth);
	applyGaussian(Iyy, Iyy_smooth);
	applyGaussian(Ixy, Ixy_smooth);

	// STEP 3: Apply Harris Operator
	harrisOperation(Ixx_smooth, Iyy_smooth, Ixy_smooth, out);

	// STEP 4: Non-Maximum suppression
	applyNonMaxSuppression(out);
}

ImageType sumOf3By3Window(Mat& in, int r, int c)
{
	ImageType result = 0;

	for (int rr = -1; rr <= 1; ++rr)
	{
		for (int cc = -1; cc <= 1; ++cc)
		{
			ImageType tmp = ((r - rr >= 0 && r - rr < in.rows)
							 && (c - cc >= 0 && c - cc < in.cols))
							 ? in.at<ImageType>(r - rr, c - cc) : 0;
			result = result + tmp;
		}
	}
	return result;
}

struct RightFeature
{
	int r; // y position in the data set
	int c; // x position in the data set
	ImageType distance; // SAD result
};

struct LeftFeature
{
	int r; // y position in the data set
	int c; // x position in the data set
	vector<RightFeature> rightMatches; // correspondences
};

void computeSAD(Mat& left, Mat& right, vector<LeftFeature>& featureSet)
{
	// traverse left
	for (int lr = 1; lr < left.rows - 1; ++lr)
	{
		for (int lc = 1; lc < left.cols - 1; ++lc)
		{
			if (left.at<ImageType>(lr, lc) != 0)
			{
				LeftFeature lFeature;
				lFeature.r = lr;
				lFeature.c = lc;
				featureSet.push_back(lFeature);
				// traverse right
				for (int rr = 1; rr < right.rows - 1; ++rr)
				{
					for (int rc = 1; rc < right.cols - 1; ++rc)
					{
						if (right.at<ImageType>(rr, rc) != 0)
						{
							ImageType diff = sumOf3By3Window(left, lr, lc) - sumOf3By3Window(right, rr, rc);
							RightFeature rFeature;
							rFeature.r = rr;
							rFeature.c = rc;
							rFeature.distance = diff;
							featureSet[featureSet.size() - 1].rightMatches.push_back(rFeature);
						}
					}
				}

			}
		}
	}
}

void sortFeatureSet(vector<LeftFeature>& featureSet)
{
	for (int i = 0; i < featureSet.size(); ++i)
	{
		sort(featureSet[i].rightMatches.begin(), featureSet[i].rightMatches.end(),
			 [](RightFeature a, RightFeature b)
		{
			return a.distance > b.distance;
		});
	}
}

void generateReport(vector<LeftFeature>& featureSet, Mat& left, Mat& right, double select=0.05)
{
	// count total number of matching in this set.
	int totalNumberOfPoints = 0;
	for (int r = 1; r < left.rows - 1; ++r)
	{
		for (int c = 1; c < left.cols - 1; ++c)
		{
			if (left.at<ImageType>(r, c) != 0)
			{
				++totalNumberOfPoints;
			}
		}
	}
	for (int r = 1; r < right.rows - 1; ++r)
	{
		for (int c = 1; c < right.cols - 1; ++c)
		{
			if (right.at<ImageType>(r, c) != 0)
			{
				++totalNumberOfPoints;
			}
		}
	}

	// accumulate correct mathcing points.
	int correct = 0;
	Mat assignmentMap = Mat::zeros(left.rows, left.cols, CV_8U);
	for (int i = 0; i < featureSet.size(); ++i)
	{
		int qualifiedSize = (int)round(featureSet[i].rightMatches.size() * select); 
		for (int j = 0; j < qualifiedSize; ++j)
		{
			assignmentMap.at<uchar>(featureSet[i].rightMatches[j].r, featureSet[i].rightMatches[j].c) = 1;
		}
	}
	// use assignment map to eliminate duplicate correct points.
	for (int r = 0; r < assignmentMap.rows; ++r)
	{
		for (int c = 0; c < assignmentMap.cols; ++c)
		{
			if (assignmentMap.at<uchar>(r, c) == 1)
			{
				++correct;
			}
		}
	}
	correct += featureSet.size();

	// generate report.
	string reportPath = "correspondences_report.txt";
	ofstream report;
	report.open(reportPath);
	report << "The total matching points in the set are: " << totalNumberOfPoints << endl
		<< "The correct corespondences number is: " << correct << endl
		<< "The incorrect corespondences number is: " << totalNumberOfPoints - correct << endl;
	report.close();
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

	//TODO:
	// STEP 1 - STEP 4
	Mat leftOut, rightOut, gtOut;
	execFeatureExtraction(leftImg, leftOut); // compute left image.
	execFeatureExtraction(rightImg, rightOut); // compute right image.

	// STEP 5 - STEP 6.
	vector<LeftFeature> featureSet;
	computeSAD(leftOut, rightOut, featureSet);
	sortFeatureSet(featureSet);
	generateReport(featureSet, leftOut, rightOut);


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