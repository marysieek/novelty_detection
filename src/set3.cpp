#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

const int LOW_THRESHOLD = 25;
const int THRESH_RATIO = 3;
const int CANNY_KERNEL_SIZE = 3;
const int BLUR_KERNEL_SIZE = 4;

void prepareImageMats (Mat &colorImage, Mat &grayImage, Mat &contourImage);
void cleanContoursWithSigma (vector <Point> &points, double maxDistanceSigma);

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

        namedWindow("COLOR");
        namedWindow("GRAY");
        namedWindow("CONTOUR");

        // analyze images
        for (auto file : fileNames) {
                cout << endl << "Analyzing file: " << file << endl;
                Mat colorImage = imread(file.c_str(), CV_LOAD_IMAGE_COLOR);
                Mat grayImage, contourImage;

                // get grayscale and contours
                prepareImageMats (colorImage, grayImage, contourImage);

                // clean from distant noise
                vector<Point> contourPoints;
                findNonZero (contourImage, contourPoints);
                cleanContoursWithSigma(contourPoints, 2.0);

                // limit image to the object
                auto boundingBox = boundingRect(contourPoints);
                colorImage = Mat(colorImage, boundingBox);
                grayImage = Mat(grayImage, boundingBox);
                contourImage = Mat(contourImage, boundingBox);

                imshow("COLOR", colorImage);
                imshow("GRAY", grayImage);
                imshow("CONTOUR", contourImage);

                waitKey(0);
                // break;
        }

        exit (0);
}

void prepareImageMats (Mat &colorImage, Mat &grayImage, Mat &contourImage) {
        cvtColor (colorImage, grayImage, CV_BGR2GRAY);
        blur (grayImage, grayImage, Size(BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE));
        morphologyEx (grayImage, grayImage, MORPH_CLOSE,
                      getStructuringElement(
                              MORPH_ELLIPSE,
                              Size(CANNY_KERNEL_SIZE+2, CANNY_KERNEL_SIZE+2)
                              )
                      );
        Canny (grayImage, contourImage, LOW_THRESHOLD, LOW_THRESHOLD*THRESH_RATIO, CANNY_KERNEL_SIZE);
}

void cleanContoursWithSigma (vector <Point> &points, double maxDistanceSigma) {

        // calculate mean
        auto sum = accumulate(points.begin(), points.end(), Point(0, 0));
        Point mean (sum.x / points.size(), sum.y / points.size());

        // calculate distances
        vector <double> pointDistances;
        pointDistances.resize (points.size());

        transform (points.begin(), points.end(), pointDistances.begin(),           [mean](Point &p) {
                return norm (p - mean);
        });

        // calculate standard deviation
        double stdDev = sqrt (accumulate (pointDistances.begin(), pointDistances.end(), 0.0, [](double x, double y) {
                return x + y*y;
        }) / pointDistances.size());

        cout << "sigma = " << stdDev << endl;

        // clean
        cout << "points before: " << points.size() << endl;
        points.erase(remove_if (points.begin(), points.end(),
                                [maxDistanceSigma, stdDev, mean](Point &p) {
                return (norm(p-mean) > maxDistanceSigma*stdDev);
        }), points.end());
        cout << "points after: " << points.size() << endl;
}
