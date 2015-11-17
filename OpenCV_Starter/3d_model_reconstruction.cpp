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
		// img[i].convertTo(img[i], DEPTH_TYPE_SIZE);
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

inline
bool isValid(Mat& point, Mat& mat, Mat& img, Mat& colorImg, std::vector<Point3d>& color)
{
	point = mat * point;
	Point2d ptOnImg;
	ptOnImg.x = point.at<DepthType>(0, 0) / point.at<DepthType>(2, 0);
	ptOnImg.y = point.at<DepthType>(1, 0) / point.at<DepthType>(2, 0);
	if (ptOnImg.x > 0 && ptOnImg.x < img.cols
		&& ptOnImg.y > 0 && ptOnImg.y < img.rows)
	{
		int cc = (int)floor(ptOnImg.x);
		int rr = (int)floor(ptOnImg.y);
		if (img.at<ImageType>(rr, cc) != 0)
		{
			Point3d tmp;
			tmp.x = colorImg.at<Vec3b>(rr, cc)[0];
			tmp.y = colorImg.at<Vec3b>(rr, cc)[1];
			tmp.z = colorImg.at<Vec3b>(rr, cc)[2];
			color.push_back(tmp);
			return true;
		}
	}
	return false;
}

bool writeToPly(std::vector<Point3d>& vec, std::vector<Point3d> color, std::string path)
{
	ofstream outFile(path.c_str());
	if (!outFile)
	{
		cerr << "Error opening output file: " << path << "!" << endl;
		return false;
	}

	outFile << "ply" << endl;
	outFile << "format ascii 1.0" << endl;
	outFile << "element vertex " << vec.size() << endl;
	outFile << "property float x" << endl;
	outFile << "property float y" << endl;
	outFile << "property float z" << endl;
	outFile << "property uchar red" << endl;
	outFile << "property uchar green" << endl;
	outFile << "property uchar blue" << endl;
	outFile << "element face 0" << endl;
	outFile << "end_header" << endl;

	////
	// Points
	////

	for (int pi = 0; pi < vec.size(); ++pi)
	{
		Point3d point = vec[pi];

		// Points
		outFile << point.x << " ";
		outFile << point.y << " ";
		outFile << point.z << " ";
		// colors
		outFile << (int)(color[pi].z) << " ";
		outFile << (int)(color[pi].y) << " ";
		outFile << (int)(color[pi].x) << " ";
		// end one line
		outFile << endl;
	}

	outFile.close();
	return true;
}

int main(int argc, char** argv)
{
	//for (int i = 0; i < 8; ++i)
	//{
	//	double k = (i & 2);
	//	std::cout << k << endl;
	//}


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

	Mat colorImg[8];
	string colorPath[8] = {
		"color/cam00_00023_0000008550.png",
		"color/cam01_00023_0000008550.png",
		"color/cam02_00023_0000008550.png",
		"color/cam03_00023_0000008550.png",
		"color/cam04_00023_0000008550.png",
		"color/cam05_00023_0000008550.png",
		"color/cam06_00023_0000008550.png",
		"color/cam07_00023_0000008550.png"

	};
	if (errorPictNum = loadMany(colorImg, colorPath, 8) != -1)
	{
		std::cerr << "Loading Silh files fiailed : (" << errorPictNum << ").\n";
		return EXIT_FAILURE;
	}

	// showMany(silhImg, silhPath, 8);
	// showMany(colorImg, colorPath, 8);

	Mat camMat[8];
	loadCamMat(camMat, 8, "cameras.txt");

	//for (int i = 0; i < 8; ++i)
	//{
	//	std::cout << camMat[i] <<endl;
	//}

	//TODO:
	// STEP 1: Generate Voxels sizes
	// STEP 2: Project to images and see if it is lit. 
	const double scale = 100;
	const double scaleBack = 1.0 / scale;
	Point2d rangeX(-2.5, 2.5), rangeY(-3.0, 3.0), rangeZ(0.0, 2.5);
	double sizeX, sizeY, sizeZ;
	std::vector<Point3d> VertResult;
	std::vector<Point3d> colorResult;

	sizeX = (rangeX.y - rangeX.x) * scale;
	sizeY = (rangeY.y - rangeY.x) * scale;
	sizeZ = (rangeZ.y - rangeZ.x) * scale;

	for (int x = 0; x < sizeX; ++x)
	{
		for (int y = 0; y < sizeY; ++y)
		{
			for (int z = 0; z < sizeZ; ++z)
			{
				// voxel real position.
				Mat voxel = (Mat_<double>(4, 1)
							 << rangeX.x + scaleBack * x
							 , rangeY.x + scaleBack * y
							 , rangeZ.x + scaleBack * z
							 , 1
							 );

				std::vector<Point3d> color;

				bool isLit = true;

				for (int i = 0; i < 8; ++i)
				{
					// Generate 8 points from one postion:
					// directtions: front - back, left - right, bottom - top 
					Mat delta = (Mat_<double>(4, 1)
								 << (i | 1) * scaleBack
								 , ((i | 2) / 2) *scaleBack
								 , ((i | 4) / 4) *scaleBack
								 , 0
								 );

					Mat real_voxel = voxel + delta;
					/*cout << real_voxel << endl;*/
					isLit = isValid(real_voxel, camMat[i], silhImg[i], colorImg[i], color);

					if (!isLit)
					{
						break;
					}
				}

				if (isLit)
				{
					Point3d finalPos;
					finalPos.x = voxel.at<DepthType>(0, 0);
					finalPos.y = voxel.at<DepthType>(1, 0);
					finalPos.z = voxel.at<DepthType>(2, 0);
					VertResult.push_back(finalPos);

					int uu = 0;
					double sum = 0.0;
					for (int u = 0; u < color.size(); ++u)
					{
						//if ( color[u].z > color[u].x)
						//{
						//	uu = u;
						//}

						double tmpColor = color[u].x + color[u].y + color[u].z;

						if (tmpColor > sum)
						{
							sum = tmpColor;
							uu = u;
						}

					}
					colorResult.push_back(color[uu]);
				}
			}
		}
	}

	writeToPly(VertResult, colorResult, "result100M2.ply");


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