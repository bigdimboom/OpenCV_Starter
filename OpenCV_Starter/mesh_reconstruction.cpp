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
#include <opencv2/flann.hpp>

using namespace std;
using namespace cv;

#define KNN 50 // the number of k nearest points
#define SIGMA 20 // mm

enum SignCount
{
	X_POSITIVE = 0,
	X_NEGTIVE,
	Y_POSITIVE,
	Y_NEGTIVE,
	Z_POSITIVE,
	Z_NEGTIVE,
	SUM_SIZE // 3 * 2 = 6
};



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

void parsePoints(string path, vector<Point3f>& pointCloud)
{
	std::string str; // Temp string to
	std::ifstream fin(path); // Open it up!
	if (fin.fail() || !fin.is_open())
	{
		cerr << "file \"" << path << "\" open error.\n";
	}

	bool isData = false;
	vector<float> tmp;

	while (fin >> str)
	{
		if (isData)
		{
			tmp.push_back(stof(str));
			if (tmp.size() == 4)
			{
				pointCloud.push_back(
					Point3f(tmp[0], tmp[1], tmp[2])
					);
				tmp.clear();
			}
		}
		else
		{
			if (str == "end_header")
			{
				isData = true;
			}
		}
	}

	fin.close(); // Close that file!
}

void readVerts(string prefix, vector<Point3f>& container)
{
	string tmp = "";
	for (int i = 1; i <= 4; ++i)
	{
		tmp = prefix + to_string(i) + ".ply";
		parsePoints(tmp, container);
		tmp.clear();
	}
}


void knnSearch(const vector<Point3f>& pointCloud,
			   Mat& indices, //(numQueries, k_dimentions, CV_32S);
			   Mat& dists //(numQueries, k_dimentions, CV_32S);
			   )
{
	Mat query = Mat(pointCloud).reshape(1);
	flann::KDTreeIndexParams indexParams;
	flann::Index kdTree(query, indexParams, cvflann::FLANN_DIST_EUCLIDEAN);
	kdTree.knnSearch(query, indices, dists, KNN);
}

void GenerateNormals(vector<Point3f>& normals, vector<Point3f>& points, Mat& idxs, Mat& dists)
{
	float factor = 1.0f / (2 * SIGMA * SIGMA);

	for (int r = 0; r < idxs.rows; ++r)
	{
		Point3f p = points[r];
		Mat covarMat(3, 3, CV_32F);
		int signCount[SignCount::SUM_SIZE] = { 0 };

		for (int c = 0; c < idxs.cols; ++c)
		{
			int index = idxs.at<int>(r, c);

			if (index == r)
			{
				continue;
			}

			Point3f pp = points[index];
			Point3f vec = pp - p;

			float d = dists.at<float>(r, c);
			float weight = exp(-(d*d) * factor);
			float denom = vec.x * vec.x + vec.y* vec.y + vec.z * vec.z;
			vec = vec / sqrt(denom); // normalize
			Mat vecMat(vec);
			Mat vecMatTranspose = vecMat.t();
			Mat result = weight * (vecMat * vecMatTranspose) * (1 / denom);
			covarMat += result;

			// for making the sign decision.
			// comparing x
			if (vec.dot(Point3f(1.0f, 0.0f, 0.0f)) >= 0.0f)
			{
				++signCount[SignCount::X_POSITIVE];
			}
			if (vec.dot(Point3f(-1.0f, 0.0f, 0.0f)) > 0.0f)
			{
				++signCount[SignCount::X_NEGTIVE];
			}
			// comparing y
			if (vec.dot(Point3f(0.0f, 1.0f, 0.0f)) >= 0.0f)
			{
				++signCount[SignCount::Y_POSITIVE];
			}
			if (vec.dot(Point3f(0.0f, -1.0f, 0.0f)) > 0.0f)
			{
				++signCount[SignCount::Y_NEGTIVE];
			}
			// comapring z
			if (vec.dot(Point3f(0.0f, 0.0f, 1.0f)) >= 0.0f)
			{
				++signCount[SignCount::Z_POSITIVE];
			}
			if (vec.dot(Point3f(0.0f, 0.0f, -1.0f)) > 0.0f)
			{
				++signCount[SignCount::Z_NEGTIVE];
			}
		}

		Mat eigenValues;
		Mat eigenVectors;
		bool success = eigen(covarMat, eigenValues, eigenVectors);
		float smallestEV = eigenValues.at<float>(0,0);
		int whichOne = 0;
		for (int rr = 1; rr < eigenValues.rows; ++rr)
		{
			if (eigenValues.at<float>(rr, 0) < smallestEV)
			{
				whichOne = rr; // pick the eiganvector that has the smallest eigenvalue.
			}
		}
		//cout << eigenValues << endl;
		//cout << eigenVectors << endl;
		Point3f finalNormal(eigenVectors.at<Vec3f>(whichOne, 0));
		//cout << finalNormal << endl;

		// Make sign decision.
		finalNormal.x = abs(finalNormal.x);
		finalNormal.y = abs(finalNormal.y);
		finalNormal.z = abs(finalNormal.z);

		if (signCount[SignCount::X_POSITIVE] < signCount[SignCount::X_NEGTIVE])
		{
			finalNormal.x = -finalNormal.x;
		}
		if (signCount[SignCount::Y_POSITIVE] < signCount[SignCount::Y_POSITIVE])
		{
			finalNormal.y = -finalNormal.y;
		}
		if (signCount[SignCount::Z_POSITIVE] < signCount[SignCount::Z_POSITIVE])
		{
			finalNormal.z = -finalNormal.z;
		}
		normals.push_back(finalNormal);
	}
}

int main(int argc, char** argv)
{
	Point3f vec(1, 2, 3);

	auto ret = vec.dot(Point3f(1.0f, 0.0, 0.0f));

	float denom = vec.x * vec.x + vec.y* vec.y + vec.z * vec.z;
	Mat vecMat(vec);
	Mat vecMatTranspose = vecMat.t();
	Mat result = 10 * (vecMat * vecMatTranspose) * (1 / denom);

	//TODO : Q1
	// STEP 1: Get K Nearest Points (K == 50)
	string applePrefix = "hw7_plys/apple_";
	string bananaPrefix = "hw7_plys/banana_";
	string lemonPrefix = "hw7_plys/lemon_";
	vector<Point3f> apple, banana, lemon;
	readVerts(applePrefix, apple);
	readVerts(bananaPrefix, banana);
	readVerts(lemonPrefix, lemon);
	Mat appleIdxs, appleDists;
	knnSearch(apple, appleIdxs, appleDists);
	//Mat bananaIdxs, bananaDists;
	//knnSearch(banana, bananaIdxs, bananaDists);
	//Mat lemonIdxs, lemonDists;
	//knnSearch(lemon, lemonIdxs, lemonDists);

	// STEP 2: USE FORMULA
	vector<Point3f> appleNormals, bananaNormals, lemonNormals;
	GenerateNormals(appleNormals, apple, appleIdxs, appleDists);


	return EXIT_SUCCESS;
}