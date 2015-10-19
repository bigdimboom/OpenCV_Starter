#include <iostream>                        // std::cout
#include <fstream>                         // std::fstream
#include <cmath>                           // std::abs
#include <opencv2/core/core.hpp>           // cv::Mat
#include <opencv2/highgui/highgui.hpp>     // cv::imread()
#include <opencv2/imgproc/imgproc.hpp>     // cv::getPerspective()

using namespace std;
using namespace cv;

#define MAX_DISPARITY 63
#define MIN_DISPARITY 0

// Compute the rank transform in n ¡Á n windows.
bool RankTransform(Mat& input, Mat& out, int windowSize)
{
	if (windowSize % 2 == 0)
	{
		return false; // windows size must be an odd number
	}

	out = Mat::zeros(input.rows, input.cols, CV_8U);

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
					if (input.at<uchar>(r + wr, c + wc) < input.at<uchar>(r, c))
					{
						++rank;
					}
				}
			}

			out.at<uchar>(r, c) = rank;
		}
	}
	return true;
}

// Compute the SAD in n ¡Á n windows.
bool SAD(Mat& inputLeft, Mat& inputRight, Mat& output, int windowSize)
{
	if (windowSize % 2 == 0)
	{
		return false; // windows size must be an odd number.
	}
	output = Mat::zeros(inputLeft.rows, inputLeft.cols, CV_8U);
	// the output disperity map.
	int center = (windowSize - 1) / 2;
	for (int r = center; r < inputLeft.rows - center; ++r)
	{
		for (int c = center + MAX_DISPARITY; c < inputLeft.cols - center; ++c)
		{
			int prevCost = INT_MAX;
			int theMin = MIN_DISPARITY;

			for (int d = 0; d >= -MAX_DISPARITY; --d) // slide window
			{
				int currentCost = 0;

				for (int wr = -center; wr < center; ++wr)
				{
					for (int wc = -center; wc < center; ++wc)
					{
						if (c - center + d >= 0)
						{
							int cost = abs(inputLeft.at<uchar>(r + wr, c + wc) -
										   inputRight.at<uchar>(r + wr, c + wc + d));
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
			output.at<uchar>(r, c) = theMin;
		}
	}
	return true;
}

// Compute the SAD in n ¡Á n windows.
bool PKRN(Mat& inputLeft, Mat& inputRight, Mat& output, Mat& assignmentMap, int windowSize = 3)
{
	if (windowSize % 2 == 0)
	{
		return false; // windows size must be an odd number.
	}

	output = Mat::zeros(inputLeft.rows, inputLeft.cols, CV_8U);
	assignmentMap = Mat::zeros(inputLeft.rows, inputLeft.cols, CV_8U);

	// the output disperity map.
	int center = (windowSize - 1) / 2;
	for (int r = center; r < inputLeft.rows - center; ++r)
	{
		for (int c = center + MAX_DISPARITY; c < inputLeft.cols - center; ++c)
		{
			int prevCost = INT_MAX;
			int prevprevCost = INT_MAX;

			int theMin = MIN_DISPARITY; 
			// int the2ndMin = MIN_DISPARITY;

			for (int d = 0; d >= -MAX_DISPARITY; --d) // slide window
			{
				int currentCost = 0; // local confidence
				for (int wr = -center; wr < center; ++wr)
				{
					for (int wc = -center; wc < center; ++wc)
					{
						if (c - center + d >= 0)
						{
							int cost = abs(inputLeft.at<uchar>(r + wr, c + wc) -
										   inputRight.at<uchar>(r + wr, c + wc + d));
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
					prevprevCost = prevCost;
					prevCost = currentCost;
					// the2ndMin = theMin;
					theMin = abs(d);
				}
			}

			if ((float)(prevprevCost ) / (float)prevCost >= 0.5f) // ( c2 / c1 >=0.5f)
			{
				output.at<uchar>(r, c) = theMin;
				assignmentMap.at<uchar>(r, c) = 1;
			}
		}
	}

	return true;
}

float ErrorRate(Mat& toBeTest, Mat& sample)
{
	int count = 0;

	for (int r = 0; r < toBeTest.rows && r < sample.rows; ++r)
	{
		for (int c = 0; c < toBeTest.rows && c < sample.rows; ++c)
		{
			float value = round((float)sample.at<uchar>(r, c) / 4.0f);
			if ( (toBeTest.at<uchar>(r, c) < (int)value - 1
				|| toBeTest.at<uchar>(r, c) > (int)value + 1) )
			{
				count++;
			}
		}
	}

	return  (float(count) / ((float)sample.rows * sample.cols));
}

float PKRNErrorRate(Mat& toBeTest, Mat& assignmentMap, Mat& sample)
{
	int count = 0;

	for (int r = 0; r < toBeTest.rows && r < sample.rows; ++r)
	{
		for (int c = 0; c < toBeTest.rows && c < sample.rows; ++c)
		{
			float value = round((float)sample.at<uchar>(r, c) / 4.0f);
			if ((toBeTest.at<uchar>(r, c) < (int)value - 1
				|| toBeTest.at<uchar>(r, c) > (int)value + 1) 
				&& assignmentMap.at<uchar>(r,c) == 1)
			{
				count++;
			}
		}
	}

	return  (float(count) / ((float)sample.rows * sample.cols));
}

int EffetivePixels(Mat& assignmentMap)
{
	int ret = 0;

	for (int r = 0; r < assignmentMap.rows;  ++r)
	{
		for (int c = 0; c < assignmentMap.rows; ++c)
		{
			if (assignmentMap.at<uchar>(r, c) == 1)
			{
				++ret;
			}
		}
	}
	return ret;
}


int main(int argc, char** argv)
{
	const char* leftPath = "teddyL.pgm";
	const char* rightPath = "teddyR.pgm";
	const char* disparityPath = "disp2.pgm";

	const char* disparityPath3by3 = "3by3.bmp";
	const char* disparityPath15by15 = "15by15.bmp";
	const char* disparityConfidence = "3by3ConfidenceDisparity.bmp";
	const char* errorRatePath = "error.txt";
	const char* PKRNerrorRatePath = "PRKNError.txt";

	//Read Img
	Mat leftImg;
	Mat rightImg;
	Mat groundTruthImg;

	Mat threeByThreeOutput;
	// disperity map generated by 3 by 3 window. 
	Mat fifteenByFifteenOutput;
	// disperity map generated by 15 by 15 window.

	leftImg = imread(leftPath, -1);
	if (!leftImg.data)
	{
		printf(" No image data Left\n ");
		return -1;
	}

	rightImg = imread(rightPath, -1);
	if (!rightImg.data)
	{
		printf(" No image data Right \n ");
		return -1;
	}
	groundTruthImg = imread(disparityPath, -1);
	if (!groundTruthImg.data)
	{
		printf(" No image data Dis\n ");
		return -1;
	}

	//namedWindow(leftPath, CV_WINDOW_AUTOSIZE);
	//imshow(disperityPath, groundTruthImg);

	// Rank Transform pass in 5 by 5 window
	Mat rankLeft, rankRight;
	bool rank1 = RankTransform(leftImg, rankLeft, 5);
	bool rank2 = RankTransform(rightImg, rankRight, 5);
	if (!rank1 || !rank2)
	{
		std::cerr << "rank compute filed\n";
		return EXIT_FAILURE;
	}
	//// SAD:
	bool result1 = SAD(rankLeft, rankRight, threeByThreeOutput, 3);
	bool result2 = SAD(rankLeft, rankRight, fifteenByFifteenOutput, 15);
	if (!result1 || !result2)
	{
		std::cerr << "CAD compute filed\n";
		return EXIT_FAILURE;
	}
	// error rate
	float errorRate1 = ErrorRate(threeByThreeOutput, groundTruthImg);
	float errorRate2 = ErrorRate(fifteenByFifteenOutput, groundTruthImg);

	ofstream errorRateFile;
	errorRateFile.open(errorRatePath);
	errorRateFile << "3 by 3's error rate: " << errorRate1 << ".\n"
		<< "15 by 15 error rate is: " << errorRate2 << ".\n";
	errorRateFile.close();

	Mat PKRNResult;
	Mat PRKNAssignment;
	bool pkrn = PKRN(rankLeft, rankRight, PKRNResult, PRKNAssignment);
	if (!pkrn)
	{
		std::cerr << "PKRN compute failed\n";
		return EXIT_FAILURE;
	}

	ofstream errorRatePKRN;
	errorRatePKRN.open(PKRNerrorRatePath);
	errorRatePKRN << "PKRN error rate: " << PKRNErrorRate(PKRNResult, PRKNAssignment, groundTruthImg) << ".\n"
		<< "the number of pixels that have been kept: " << EffetivePixels(PRKNAssignment) << ".\n";
	errorRatePKRN.close();

	namedWindow(disparityPath3by3, CV_WINDOW_AUTOSIZE);
	namedWindow(disparityPath15by15, CV_WINDOW_AUTOSIZE);
	namedWindow("PKRN", CV_WINDOW_AUTOSIZE);
	imshow(disparityPath3by3, threeByThreeOutput);
	imshow(disparityPath15by15, fifteenByFifteenOutput);
	imshow("PKRN", PKRNResult);

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
		succ = imwrite(disparityPath3by3, threeByThreeOutput);
		succ = imwrite(disparityPath15by15, fifteenByFifteenOutput);
		succ = imwrite(disparityConfidence, PKRNResult);
		if (!succ)
		{
			printf(" Image writing fialed \n ");
			return -1;
		}
	}

	return 0;
}