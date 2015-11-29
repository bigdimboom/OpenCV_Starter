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

#define FOCAL_LENGTH 1247 // pixels
#define UNIT_BASELINE 40  // millimeters

#define DEPTH_TYPE_SIZE CV_64F
#define DEPTH_INFINITY 999999
typedef double DepthType;

#define IMAGE_TYPE_SIZE CV_8U
#define MAX_COLOR 255
#define MIN_COLOR 0
typedef uchar ImageType;

bool loadImage(Mat& img, string path, int type)
{
	img = imread(path, type);
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

void generateVerts(vector<Point3f>& outputVerts,
				   Mat& depth,
				   vector<Point3i>& outputColors,
				   Mat& texture,
				   vector<bool>& flags
				   )
{
	for (int r = 0; r < depth.rows; ++r)
	{
		for (int c = 0; c < depth.cols; ++c)
		{
			bool flag = true;
			float x = (float)c - depth.cols * 0.5f;
			float y = (float)r - depth.rows * 0.5f;
			float z = depth.at<ImageType>(r, c);

			// reverse fomula for assignment 4 : visualize depth map
			//z = z * (1.0f/3.0f) * z / (FOCAL_LENGTH * 4.0f * UNIT_BASELINE);

			uchar B = texture.at<Vec3b>(r, c)[0];
			uchar G = texture.at<Vec3b>(r, c)[1];
			uchar R = texture.at<Vec3b>(r, c)[2];

			if (z == 0.0f) // mark out points that are valid;
			{
				flag = false;
			}

			outputVerts.push_back(Point3f(x, y, z));
			outputColors.push_back(Point3i(R, G, B));
			flags.push_back(flag);
		}
	}
}

void generateIndices(vector<int>& out, vector<bool>& flags, Mat& depth)
{
	int width = depth.cols;
	for (int r = 0; r < depth.rows - 1; ++r)
	{
		for (int c = 0; c < depth.cols - 1; ++c)
		{

			bool marks[4];

			marks[0] = flags[r * width + c];
			marks[1] = flags[r * width + c + 1];
			marks[2] = flags[r * width + c + 1 + width];
			marks[3] = flags[r * width + c + width];

			if (marks[0] && marks[1] && marks[2]) // only write down valid points
			{
				out.push_back(r * width + c);
				out.push_back(r * width + c + 1);
				out.push_back(r * width + c + 1 + width);
			}

			if (marks[0] && marks[2] && marks[3]) // only write down valid points
			{
				out.push_back(r * width + c);
				out.push_back(r * width + c + 1 + width);
				out.push_back(r * width + c + width);
			}
		}
	}
}

void writeToPly(string path, vector<Point3f>& verts, vector<Point3i>& colors, vector<int>& indices)
{
	ofstream outFile(path);
	if (outFile.fail() || !outFile.is_open())
	{
		cerr << "Error opening output file: " << path << "!" << endl;
	}

	outFile << "ply" << endl;
	outFile << "format ascii 1.0" << endl;
	outFile << "element vertex " << verts.size() << endl;
	outFile << "property float x" << endl;
	outFile << "property float y" << endl;
	outFile << "property float z" << endl;
	outFile << "property uchar red" << endl;
	outFile << "property uchar green" << endl;
	outFile << "property uchar blue" << endl;
	outFile << "element face " << indices.size() / 3 << endl;
	outFile << "property list uchar int vertex_indices" << endl;
	outFile << "end_header" << endl;

	////
	// Points
	////

	for (int pi = 0; pi < verts.size(); ++pi)
	{
		Point3d point = verts[pi];

		// Points
		outFile << point.x << " ";
		outFile << point.y << " ";
		outFile << point.z << " ";
		// colors
		outFile << (colors[pi].x) << " ";
		outFile << (colors[pi].y) << " ";
		outFile << (colors[pi].z) << " ";
		// end one line
		outFile << endl;
	}

	for (int pi = 0; pi < indices.size(); pi += 3)
	{
		outFile << 3 << " ";
		outFile << indices[pi] << " ";
		outFile << indices[pi + 1] << " ";
		outFile << indices[pi + 2] << " ";
	}

	outFile.close();
}

int main(int argc, char** argv)
{
	Mat depth, texture;
	string depthInputPath = "depth.bmp";
	string texturePath = "view3.png";
	loadImage(depth, depthInputPath, -1);
	loadImage(texture, texturePath, CV_LOAD_IMAGE_COLOR);

	vector<Point3f> grid;
	vector<int> indices;
	vector<Point3i> colors;
	vector<bool> flags;

	generateVerts(grid, depth, colors, texture, flags);
	generateIndices(indices, flags, depth);

	writeToPly("mesh_output.ply", grid, colors, indices);

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