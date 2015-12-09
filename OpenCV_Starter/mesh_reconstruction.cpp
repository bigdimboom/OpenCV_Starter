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
#include <random>

using namespace std;
using namespace cv;

#define KNN 50 // the number of k nearest points
#define SIGMA 20 // scale : mm
#define NUM_OF_BINS 11 // bins: 11 x 11
#define BIN_SIZE 3 // 3 x 3 mm


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

void readBatchVerts(string prefix, vector<Point3f>* containers, int size)
{
	string tmp = "";
	for (int i = 1; i <= 4; ++i)
	{
		tmp = prefix + to_string(i) + ".ply";
		parsePoints(tmp, containers[i - 1]);
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

void knnBatchSearch(vector<Point3f>* points,
					Mat* indices,
					Mat* dists,
					int size)
{
	for (int i = 0; i < size; ++i)
	{
		knnSearch(points[i], indices[i], dists[i]);
	}
}

void GenerateNormals(vector<Point3f>& normals,
					 vector<Point3f>& points,
					 Mat& idxs,
					 Mat& dists)
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
			float denom = vec.dot(vec);
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
		float smallestEV = eigenValues.at<float>(0, 0);
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

		finalNormal = finalNormal / (sqrt(finalNormal.dot(finalNormal)));

		normals.push_back(finalNormal);
	}
}

void GenBatchNormals(vector<Point3f>* normals,
					 vector<Point3f>* points,
					 Mat* idxs,
					 Mat* dists,
					 int size)
{
	for (int i = 0; i < size; ++i)
	{
		GenerateNormals(normals[i], points[i], idxs[i], dists[i]);
	}
}

void printResult(vector<Point3f>& points,
				 vector<Point3f>& normals,
				 string fileName)
{
	ofstream outFile(fileName);
	if (!outFile.is_open() || outFile.fail())
	{
		cerr << "Error opening output file: " << fileName << "!" << endl;
	}

	for (int pi = 0; pi < points.size(); ++pi)
	{
		Point3d point = points[pi];
		Point3d normal = normals[pi];
		// Points
		outFile << point.x << " ";
		outFile << point.y << " ";
		outFile << point.z << " ";
		// normals
		outFile << normal.x << " ";
		outFile << normal.y << " ";
		outFile << normal.z << " ";
		// end one line
		outFile << endl;
	}

	outFile.close();
}

void spinningImage(const Point3f & point, const Point3f& normal, vector<Point3f>& pointCloud, Mat& image)
{
	int size = BIN_SIZE * NUM_OF_BINS;
	image = Mat::zeros(size + 1, size + 1, CV_8U);
	for (int i = 0; i < pointCloud.size(); ++i)
	{
		// Project to 2D : from cartesian to cylindrical
		Point3f target = pointCloud[i];
		Point3f vec = target - point; // the slop of the rect triangle
		float newY = normal.dot(vec); // beta
		float newX = sqrt(vec.dot(vec) - newY * newY); // alpha

		// translate to coresponding image position
		int y = abs((int)(floor(size * 0.5f - newY)));
		int x = abs((int)(floor(size * 0.5f - newX)));

		if (x <= size && y <= size)
		{
			++image.at<uchar>(y, x);
		}
	}
}

void batchSpinningImage(vector<Point3f>& points,
				   vector<Point3f>& normals,
				   int numOfSelected,
				   map<int, Mat>& images)
{
	default_random_engine generator;
	uniform_int_distribution<int> distribution(0, (int)points.size() - 1);
	for (int count = 0; count < numOfSelected; ++count)
	{
		int selected = distribution(generator);

		auto hit = images.find(selected);

		if (hit != images.end())
		{
			continue;
		}

		Point3f pt = points[selected];
		Point3f n = normals[selected];
		spinningImage(pt, n, points, images[selected]);
		//Mat tmp = images[selected];
		//bitwise_not(tmp, tmp);
		// OR: Mat invSrc =  cv::Scalar::all(255) - src;
		//equalizeHist(tmp, tmp);
	}
}

void printImgs(map<int, Mat>& images, vector<Point3f>& point, string tag)
{
	for (const auto & img : images)
	{
		int index = img.first;
		Mat theImg = img.second.clone();
		bitwise_not(theImg, theImg);
		equalizeHist(theImg, theImg);
		bool succ = imwrite("image_output/" + tag + "_"
							+ "cood_"
							+ to_string(point[index].x) + "_"
							+ to_string(point[index].y) + "_"
							+ to_string(point[index].z) + ".bmp"
							, theImg);
		if (!succ)
		{
			printf(" Image writing fialed \n ");
		}
	}
}

int main(int argc, char** argv)
{
	//TODO : Q1
	// STEP 1: Get K Nearest Points (K == 50)
	vector<Point3f> apple[4], banana[4], lemon[4];
	string applePrefix = "hw7_plys/apple_";
	string bananaPrefix = "hw7_plys/banana_";
	string lemonPrefix = "hw7_plys/lemon_";
	readBatchVerts(applePrefix, apple, 4);
	readBatchVerts(bananaPrefix, banana, 4);
	readBatchVerts(lemonPrefix, lemon, 4);

	Mat appleIdxs[4], appleDists[4];
	knnBatchSearch(apple, appleIdxs, appleDists, 4);
	Mat bananaIdxs[4], bananaDists[4];
	knnBatchSearch(banana, bananaIdxs, bananaDists, 4);
	Mat lemonIdxs[4], lemonDists[4];
	knnBatchSearch(lemon, lemonIdxs, lemonDists, 4);

	// STEP 2: USE FORMULA
	vector<Point3f> appleNormals[4], bananaNormals[4], lemonNormals[4];
	GenBatchNormals(appleNormals, apple, appleIdxs, appleDists, 4);
	GenBatchNormals(bananaNormals, banana, bananaIdxs, bananaDists, 4);
	GenBatchNormals(lemonNormals, lemon, lemonIdxs, lemonDists, 4);
	//printResult(apple[0], appleNormals[0], "apple_1_result.txt");
	//printResult(banana[0], bananaNormals[0], "banana_1_result.txt");
	//printResult(lemon[0], lemonNormals[0], "lemon_1_result.txt");


	// TODO: Q2 "Spinning Image Recognition"
	map<int, Mat> appleImgs[2], bananaImgs[2], lemonImgs[2];
	for (int i = 0; i < 2; ++i)
	{
		batchSpinningImage(apple[i], appleNormals[i], 30, appleImgs[i]);
		batchSpinningImage(banana[i], bananaNormals[i], 30, bananaImgs[i]);
		batchSpinningImage(lemon[i], lemonNormals[i], 30, lemonImgs[i]);
		//printImgs(appleImgs[i], apple[i], "apple_" + to_string(i + 1));
		//printImgs(bananaImgs[i], banana[i], "banana_" + to_string(i + 1));
		//printImgs(lemonImgs[i], lemon[i], "lemon_" + to_string(i + 1));
	}

	// TODO: Q3 "Train data and classify data"





	return EXIT_SUCCESS;
}