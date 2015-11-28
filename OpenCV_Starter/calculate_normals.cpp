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

using namespace std;
using namespace cv;

const int kVertSize = 10002;

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

void readVertAndIdxsFromPLY(string path, vector<Point3f>& verts, vector<int>& idxs)
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
			if (verts.size() == kVertSize)
			{
				// start put in index
				idxs.push_back(stoi(str));
			}
			else
			{
				tmp.push_back(stof(str));

				if (tmp.size() == 5)
				{
					verts.push_back(Point3f(tmp[0], tmp[1], tmp[2]));
					tmp.clear();
				}
			}
		}
		else if (str == "end_header")
		{
			isData = true;
		}
	}

	fin.close(); // Close that file!
}

void calculateNormals(vector<Point3f>& verts, vector<int>& idx, vector<Point3f>& output)
{
	int indexSize = idx.size();
	int vertsSize = verts.size();

	vector<Point3f>* normalsPtr = new vector<Point3f>[vertsSize];

	for (int i = 0; i < indexSize - 3; i += 3)
	{
		Point3f a = verts[idx[i]];
		Point3f b = verts[idx[i + 1]];
		Point3f c = verts[idx[i + 2]];

		Point3f vec1 = b - a;
		Point3f vec2 = c - b;

		Point3f normal = vec1.cross(vec2); // un-nomalized

		normalsPtr[idx[i]].push_back(normal);
		normalsPtr[idx[i + 1]].push_back(normal);
		normalsPtr[idx[i + 2]].push_back(normal);
	}

	for (int i = 0; i < vertsSize; ++i)
	{
		Point3f tmp(0.0f, 0.0f, 0.0f);

		unsigned int num = normalsPtr[i].size();

		output.push_back(tmp);

		if (num)
		{
			for (const auto & elem : normalsPtr[i])
			{
				tmp += elem;
			}

			// average the normals
			tmp.x /= num;
			tmp.y /= num;
			tmp.z /= num;

			float len = sqrt(tmp.x * tmp.x + tmp.y * tmp.y + tmp.z * tmp.z);

			// nomalized the normals
			tmp.x /= len;
			tmp.y /= len;
			tmp.z /= len; 

			output[i] = tmp;
		}
	}

	delete[] normalsPtr;
}

bool writePLY(string outPath, vector<Point3f>& verts, vector<Point3f>& normals, vector<int>& indices)
{
	ofstream outFile(outPath);
	if (!outFile.is_open() || outFile.fail())
	{
		cerr << "Error opening output file: " << outPath << "!" << endl;
		return false;
	}

	//outFile << "ply" << endl;
	//outFile << "format ascii 1.0" << endl;
	//outFile << "comment" << endl;
	//outFile << "element vertex " << verts.size() << endl;
	//outFile << "property float x" << endl;
	//outFile << "property float y" << endl;
	//outFile << "property float z" << endl;
	//outFile << "property float nx" << endl;
	//outFile << "property float ny" << endl;
	//outFile << "property float nz" << endl;
	//outFile << "element face 0" << endl;
	//outFile << "end_header" << endl;

	////
	// Points
	////

	for (int pi = 0; pi < verts.size(); ++pi)
	{
		Point3d point = verts[pi];
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
	return true;
}

int main(int argc, char** argv)
{
	vector<Point3f> verts;
	vector<int> indices;
	vector<Point3f> avgNormals;

	readVertAndIdxsFromPLY("gargoyle.ply", verts, indices);
	calculateNormals(verts, indices, avgNormals);
	writePLY("normals_result.txt", verts, avgNormals, indices);

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