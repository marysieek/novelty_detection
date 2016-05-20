#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

const double APPROXPOLYDP_EPS = 25.0;

int main (int argc, char** argv) {

        // check argument
        if (argc < 2) {
                cerr << "No directory given" << endl;
                exit (-1);
        }

        // load file names
        vector<string> fileNames;
        glob((string(argv[1]) + "/*.jpg"), fileNames, false);
        auto numberOfFiles = fileNames.size();

        cout << "Found " << numberOfFiles << " files in " << argv[1] << endl;

        namedWindow("PROBLEM");

        // counters
        unsigned quadsFound = 0;
        unsigned trianglesFound = 0;
        unsigned unknownsFound = 0;

        // analyze images
        for (auto file : fileNames) {
                cout << endl << "Analyzing file: " << file << endl;
                Mat image = imread(file.c_str(), CV_LOAD_IMAGE_GRAYSCALE);

                blur(image, image, Size(3, 3), Point(-1, -1));

                // invert image
                threshold (image, image, 127, 255, THRESH_BINARY_INV);

                // finding contours
                vector< vector<Point> > contours;
                findContours (image, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

                if (contours.size() < 1) {
                        cout << "No contours found - skipping" << endl;
                        break;
                }

                cout << "\tFound " << contours.size() << (contours.size() == 1 ? " contour" : " contours - WARNING: there should be only one contour per image") << endl;

                // counting edges
                vector<Point> polygon (contours.size());
                approxPolyDP (contours[0], polygon, APPROXPOLYDP_EPS, true);

                auto numberOfSides = polygon.size();

                // get figure type
                string figureName = "UNKNOWN";
                switch (numberOfSides) {
                case 3:
                        figureName = "triangle";
                        ++trianglesFound;
                        break;
                case 4:
                        figureName = "quadrilateral";
                        ++quadsFound;
                        break;
                default:
                        ++unknownsFound;
                }

                cout << "\tShape found: " << figureName << endl;
        }

        cout << endl << "Statistics: " << endl;
        cout << "\tTotal images: " << numberOfFiles << endl;
        cout << "\tTriangles:    " << trianglesFound << endl;
        cout << "\tQuads:        " << quadsFound << endl;
        cout << "\tUnknowns:     " << unknownsFound << endl;

        exit (0);
}
