#include <iostream>                        // std::cout
#include <opencv2/core/core.hpp>           // cv::Mat
#include <opencv2/highgui/highgui.hpp>     // cv::imread()
#include <opencv2/imgproc/imgproc.hpp>     // cv::getPerspective()

using namespace std;
using namespace cv;

#define TARGET_ROW 500 // the row size of target frame
#define TARGET_COL 940 // the col size of target frame

vector<Point2f> gDistortPts; // Hand Picked on Sample 
vector<Point2f> gTargetPts; // The output

void InitPickPoints()
{
	// 4 distored points
	gDistortPts.push_back(Point2f(22, 193)); // left bottom
	gDistortPts.push_back(Point2f(246, 50)); // left top
	gDistortPts.push_back(Point2f(402, 74)); // right top
	gDistortPts.push_back(Point2f(278, 279)); // right bottom
}

void InitOutputPts()
{
	// these will be the parallel plane vector of point 
	// 4 None distorted point
	gTargetPts.push_back(Point2f(0, 0));
	gTargetPts.push_back(Point2f(TARGET_COL - 1, 0));
	gTargetPts.push_back(Point2f(TARGET_COL - 1, TARGET_ROW - 1));
	gTargetPts.push_back(Point2f(0, TARGET_ROW - 1));
}

Mat& GetProjMat(const Point2f src[], const Point2f dst[])
{
	Mat tansformMat = Mat(3, 3, CV_32FC1);

	return tansformMat;
}

// Using home made transform function
void ProcessImg(Mat& src, Mat& dest)
{
	// TODO:
}

// Using OpenCV built-in
void ProcessImgCV(Mat& src, Mat& dest)
{
	//TODO:
	Mat transformationMatrix = getPerspectiveTransform(&gTargetPts[0], &gDistortPts[0]);
	warpPerspective(src, dest, transformationMatrix.inv(), dest.size(), CV_INTER_LINEAR, BORDER_ISOLATED);
}

int main(int argc, char** argv)
{
	const char* inputPath = "basketball-court.ppm";
	const char* outputPath = "out.bmp";

	//Read Img
	Mat inputImg;
	inputImg = imread(inputPath, -1);

	if (!inputImg.data)
	{
		printf(" No image data \n ");
		return -1;
	}

	// Init Mapping Points
	InitPickPoints();
	InitOutputPts();

	// Draw Clip 
	vector<Point> not_a_rect_shape;
	not_a_rect_shape.push_back(Point(22, 193)); // left bottom
	not_a_rect_shape.push_back(Point(246, 50)); // left top
	not_a_rect_shape.push_back(Point(402, 74)); // right top
	not_a_rect_shape.push_back(Point(278, 279)); // right bottom

	const Point* point = &not_a_rect_shape[0];
	int n = (int)not_a_rect_shape.size();
	polylines(inputImg, &point, &n, 1, true, Scalar(0, 255, 0), 3, CV_AA);
	namedWindow("Clip", CV_WINDOW_AUTOSIZE);
	imshow("Clip", inputImg);

	// Where you output
	Mat outputImg;
	outputImg = Mat::zeros(TARGET_ROW, TARGET_COL, CV_8UC3);

	// Applay Processing Function
	// TODO
	//ProcessImgCV(inputImg, outputImg);
	//ProcessImgCV(inputImg, outputImg);
	ProcessImgCV(inputImg, outputImg);

	namedWindow(outputPath, CV_WINDOW_AUTOSIZE);
	imshow(outputPath, outputImg);
	std::cout << "Press \'s\' to save, \'Esc'\ to close the program.\n";
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
		succ = imwrite(outputPath, outputImg);
		if (!succ)
		{
			printf(" Image writing fialed \n ");
			return -1;
		}
	}

	return 0;
}