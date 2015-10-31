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
	gDistortPts.push_back(Point2f(22, 193)); // top left
	gDistortPts.push_back(Point2f(278, 279)); // bottom left
	gDistortPts.push_back(Point2f(402, 74)); // bottom right
	gDistortPts.push_back(Point2f(246, 50)); // top right

}

void InitOutputPts()
{
	// these will be the parallel plane vector of point 
	// 4 None distorted points
	gTargetPts.push_back(Point2f(0, 0));
	gTargetPts.push_back(Point2f(0, TARGET_ROW - 1));
	gTargetPts.push_back(Point2f(TARGET_COL - 1, TARGET_ROW - 1));
	gTargetPts.push_back(Point2f(TARGET_COL - 1, 0));
}

Mat GetProjMat(const Point2f src[], int targetRowSize, int targetColSize)
{
	Mat X = ( Mat_<float>(8, 1) );
	Mat right = (Mat_<float>(8, 1) << 22, 278, 402, 246, 193, 279, 74, 50);
	Mat left;

	for (int i = 0; i < 4; ++i)
	{
		Mat row = (
			Mat_<float>(1, 8)
			<< gTargetPts[i].x, gTargetPts[i].y, 1,
			0, 0, 0,
			-gTargetPts[i].x * gDistortPts[i].x, -gTargetPts[i].y * gDistortPts[i].x
			);
		left.push_back(row);
	}

	for (int i = 0; i < 4; ++i)
	{
		Mat row = (
			Mat_<float>(1, 8)
			<< 0, 0, 0,
			gTargetPts[i].x, gTargetPts[i].y, 1,
			-gTargetPts[i].x * gDistortPts[i].y, -gTargetPts[i].y * gDistortPts[i].y
			);
		left.push_back(row);
	}

	//std::cout << left << endl;
	//std::cout << right << endl;

	solve(left, right, X, DECOMP_LU);

	std::cout << X << endl;

	Mat ret = (Mat_<float>(3, 3) <<
			   X.at<float>(0, 0),
			   X.at<float>(1, 0),
			   X.at<float>(2, 0),
			   X.at<float>(3, 0),
			   X.at<float>(4, 0),
			   X.at<float>(5, 0),
			   X.at<float>(6, 0),
			   X.at<float>(7, 0),
			   1
			   );

	// std::cout << ret;

	return ret;
}

// Using home made transform function
void ProcessImg(Mat& src, Mat& dest)
{
	// TODO:
	// Q1:
	Mat_<Vec3b> _src = src;
	Mat_<Vec3b> _dest = dest;

	Mat transformationMatrix = GetProjMat(&gDistortPts[0], TARGET_ROW, TARGET_COL);
	//transformationMatrix.convertTo(transformationMatrix, CV_64F);
	//warpPerspective(src, dest, transformationMatrix.inv(), dest.size(), CV_INTER_LINEAR, BORDER_ISOLATED);

	for (int i = 0; i < src.cols; ++i)
	{
		for (int j = 0; j < src.rows; ++j)
		{
			int x = i, y = j;

			Mat orign = Mat(Point3f(i, j, 1));
			Mat ret = transformationMatrix.inv() * orign; // scale back
			Point3f retPts(ret);

			x =(int) retPts.x / retPts.z;
			y = (int) retPts.y / retPts.z;

			if (x >= 0 && x < dest.cols && y >= 0 && y < dest.rows)
			{
				_dest(y, x) = _src(j, i);
			}
		}
	}

	// Q2: filling in the "holes" resampling or backward mapping

	for (int i = 0; i < dest.cols; ++i)
	{
		for (int j = 0; j < dest.rows; ++j)
		{

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

	// Applay Processing Function
	// TODO
	// ProcessImgCV(inputImg, outputImg);
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