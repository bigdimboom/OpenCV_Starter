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

Mat GetProjMat(const Point2f src[], int targetRowSize, int targetColSize)
{
	float dx1 = src[1].x - src[2].x; //
	float dy1 = src[1].y - src[2].y; //

	float dx2 = src[3].x - src[2].x; //
	float dy2 = src[3].y - src[2].y; //

	float ZGMx = src[0].x - src[1].x + src[2].x - src[3].x; //
	float ZGMy = src[0].y - src[1].y + src[2].y - src[3].y; //

	float g = (ZGMx * dy2 - ZGMy * dx2) / (dx1 * dy2 - dy1 * dx2); //
	float h = (ZGMy * dx1 - ZGMx * dy1) / (dx1 * dy2 - dy1 * dx2); //

	float a = src[1].x - src[0].x + g * src[1].x; //
	float b = src[3].x - src[0].x + h * src[3].x; //
	float c = src[0].x; //
	float d = src[1].y - src[0].y + g * src[1].y; //
	float e = src[3].y - src[0].y + h * src[3].y; //
	float f = src[0].y;

	Mat C = (Mat_<float>(3, 3) << a, b, c, d, e, f, g, h, 1);

	//cout << C << endl;

	Mat Scale = (Mat_<float>(3, 3) << targetColSize, 0, 0, 0, targetRowSize, 0, 0, 0, 1);

	Mat ret = C * Scale;

	return ret;
}

// Using home made transform function
void ProcessImg(Mat& src, Mat& dest)
{
	// TODO:
	Mat_<Vec3b> _src = src;
	Mat_<Vec3b> _dest = dest;

	Mat transformationMatrix = GetProjMat(&gDistortPts[0], TARGET_ROW, TARGET_COL);

	for (int i = 0; i < src.rows; ++i)
	{
		for (int j = 0; j < src.cols; ++j)
		{
			int x = i, y = j;

			Mat orign = Mat(Point3f(i, j, 1));
			Mat ret = transformationMatrix * orign;
			Point3f retPts(ret);
			x = retPts.x / retPts.z;
			y = retPts.y / retPts.z;
			if (x >= 0 && x < dest.rows && y >= 0 && y < dest.cols)
			{
				_dest(x, y) = _src(i, j);
			}
		}
	}
}

// Using OpenCV built-in
void ProcessImgCV(Mat& src, Mat& dest)
{
	//TODO:
	Mat transformationMatrix = getPerspectiveTransform(&gDistortPts[0], &gTargetPts[0]);
	warpPerspective(src, dest, transformationMatrix, dest.size(), CV_INTER_LINEAR, BORDER_ISOLATED);
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
	//outputImg = Mat::zeros(1, 1, CV_8UC3);

	// Applay Processing Function
	// TODO
	//ProcessImgCV(inputImg, outputImg);
	ProcessImg(inputImg, outputImg);

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