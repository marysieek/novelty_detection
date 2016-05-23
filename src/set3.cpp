#include <algorithm>
#include <cmath>
#include <iomanip>
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
const float ROUNDNESS_LIMIT_MARGIN = 0.05;
const float ROUNDNESS_PASS_LIMIT = 0.80;
const int HUE_HIST_BINS = 23;
const float HUE_HIST_MIN_CORREL = 0.87;
const int HUE_HIST_MIN_MATCHES = 4;
const float WEIGHT_ROUNDNESS = 0.2;
const float WEIGHT_HUE_HIST = 0.8;

typedef struct {
        string fileName;
        double roundness;
        bool roundnessPass;
        vector<double> hu;
        Mat hueImage;
        Mat hueHist;
        bool hueHistPass;
        float score;
} ObjectData;

vector<ObjectData> analyzeImages (vector<string> &fileNames);
void compareRoundness(vector<ObjectData> training, vector<ObjectData> testing);
void compareHueHistograms(vector<ObjectData> training, vector<ObjectData> testing);
void prepareImageMats(Mat &colorImage, Mat &grayImage, Mat &contourImage);
void cleanContoursWithSigma(vector<Point> &points, double maxDistanceSigma);
void printData(ObjectData &data);
void calculateScore(ObjectData &data);

int main(int argc, char **argv) {

        // check argument
        if (argc < 3) {
                cerr << "No enough directories given" << endl;
                exit(-1);
        }
        string trainingFilesPattern = string(argv[1]) + "/*.jpg";
        string testFilesPattern = string(argv[2]) + "/*.jpg";

        // load file names
        vector<string> trainingFileNames, testingFileNames;
        glob(trainingFilesPattern, trainingFileNames, false);
        glob(testFilesPattern, testingFileNames, false);
        auto numberOfFiles = trainingFileNames.size();

        cout << "Found " << numberOfFiles << " files in " << argv[1] << endl;

        namedWindow("COLOR");
        namedWindow("GRAY");
        namedWindow("CONTOUR");
        namedWindow("HUE");

        auto trainingData = analyzeImages(trainingFileNames);
        auto testingData = analyzeImages(testingFileNames);
        compareRoundness(trainingData, testingData);
        compareHueHistograms(trainingData, testingData);

        // for (auto object : testingData) {
        //         calculateScore(object);
        //         cout << "File: " << object.fileName << endl;
        //         cout << "\tScore: " << object.score << endl;
        // }

        exit(0);
}

vector<ObjectData> analyzeImages (vector<string> &fileNames) {
        vector<ObjectData> dataBuffer;
        for (auto file : fileNames) {
                cout << endl << "Analyzing file: " << file << endl;
                Mat colorImage = imread(file.c_str(), CV_LOAD_IMAGE_COLOR);
                Mat grayImage, contourImage;
                ObjectData data;

                data.fileName = file;

                // get grayscale and contours
                prepareImageMats(colorImage, grayImage, contourImage);

                // clean from distant noise
                vector<Point> contourPoints;
                findNonZero(contourImage, contourPoints);
                cleanContoursWithSigma(contourPoints, 2.0);

                // limit image to the object
                auto boundingBox = boundingRect(contourPoints);
                colorImage = Mat(colorImage, boundingBox);
                grayImage = Mat(grayImage, boundingBox);
                contourImage = Mat(contourImage, boundingBox);

                // calculate shape coefficients

                // calculate contour moments

                // calculate hue histograms
                Mat hsvImage;
                Mat hsvPlanes[3];

                cvtColor(colorImage, hsvImage, CV_BGR2HSV);
                split(hsvImage, hsvPlanes);
                data.hueImage = hsvPlanes[1];

                int hueHistSize = HUE_HIST_BINS;
                float range [] = {0, 180};
                const float *hueHistRange = range;
                calcHist(&data.hueImage, 1, 0, Mat(), data.hueHist, 1, &hueHistSize, &hueHistRange, 1, 0);

                // debug print
                imshow("COLOR", colorImage);
                imshow("GRAY", grayImage);
                imshow("CONTOUR", contourImage);
                imshow("HUE", data.hueImage);
                // printData(data);

                // backlog data
                dataBuffer.push_back(data);

                // waitKey(0);
                // break;
        }
        return dataBuffer;
}

void compareRoundness(vector<ObjectData> training, vector<ObjectData> testing) {
        cout << "Roundness tests" << endl;
        double minRoundness = min_element(
                training.begin(),
                training.end(),
                [](ObjectData o1, ObjectData o2) {
                return o1.roundness < o2.roundness;
        })->roundness;
        unsigned passed = 0;
        unsigned failed = 0;

        cout << "\tminimal roundness found in training: " << minRoundness << endl;
        minRoundness *= 1.0 - ROUNDNESS_LIMIT_MARGIN;
        cout << "\troundness lower limit for pass: " << minRoundness << endl;

        for (auto object : testing) {
                if (object.roundness > minRoundness) {
                        passed++;
                        object.roundnessPass = true;
                } else {
                        failed++;
                        object.roundnessPass = false;
                }
        }

        cout << "passed : " << passed << endl;
        cout << "failed : " << failed << endl;

}

void compareHueHistograms(vector<ObjectData> training, vector<ObjectData> testing) {
        int passed = 0;
        int failed = 0;
        for (auto testObject : testing) {
                int matches = 0;
                for (auto trainingObject : training) {
                        auto comparisonResult = compareHist(testObject.hueHist, trainingObject.hueHist, CV_COMP_CORREL);
                        if (comparisonResult >= HUE_HIST_MIN_CORREL) {
                                matches++;
                        }
                }
                cout << "File: " << testObject.fileName << endl;
                cout << "\tMatches : " << matches << endl;

                if (matches >= HUE_HIST_MIN_MATCHES) {
                        passed++;
                        testObject.hueHistPass = true;
                } else {
                        failed++;
                        testObject.hueHistPass = false;
                }
        }

        cout << "passed : " << passed << endl;
        cout << "failed : " << failed << endl;
}

void prepareImageMats(Mat &colorImage, Mat &grayImage, Mat &contourImage) {
        cvtColor(colorImage, grayImage, CV_BGR2GRAY);
        blur(grayImage, grayImage, Size(BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE));
        morphologyEx(
                grayImage, grayImage, MORPH_CLOSE,
                getStructuringElement(
                        MORPH_ELLIPSE, Size(CANNY_KERNEL_SIZE + 2, CANNY_KERNEL_SIZE + 2)));
        Canny(grayImage, contourImage, LOW_THRESHOLD, LOW_THRESHOLD * THRESH_RATIO,
              CANNY_KERNEL_SIZE);
}

void cleanContoursWithSigma(vector<Point> &points, double maxDistanceSigma) {

        // calculate mean
        auto sum = accumulate(points.begin(), points.end(), Point(0, 0));
        Point mean(sum.x / points.size(), sum.y / points.size());

        // calculate distances
        vector<double> pointDistances;
        pointDistances.resize(points.size());

        transform(points.begin(), points.end(), pointDistances.begin(),
                  [mean](Point &p) {
                return norm(p - mean);
        });

        // calculate standard deviation
        double stdDev =
                sqrt(accumulate(pointDistances.begin(), pointDistances.end(), 0.0,
                                [](double x, double y) {
                return x + y * y;
        }) /
                     pointDistances.size());

        // cout << "sigma = " << stdDev << endl;

        // clean
        // cout << "points before: " << points.size() << endl;
        points.erase(remove_if(points.begin(), points.end(),
                               [maxDistanceSigma, stdDev, mean](Point &p) {
                return (norm(p - mean) > maxDistanceSigma * stdDev);
        }),
                     points.end());
        // cout << "points after: " << points.size() << endl;
}

void printData(ObjectData &data) {
        unsigned wsPad = 12;
        auto printKey = [wsPad](string key) {
                                cout << setfill(' ') << setw(wsPad) << key << " : ";
                        };
        cout << "ObjectData fields:" << endl;
        printKey("filename");
        cout << data.fileName << endl;
        printKey("roundness");
        cout << data.roundness << endl;
        printKey("hu");
        cout << "values:" << endl;
        string huFill(' ', wsPad);
        for (auto h : data.hu) {
                printKey("");
                cout << h << endl;
        }
}

void calculateScore(ObjectData &data) {
        data.score = WEIGHT_ROUNDNESS * (int)data.roundnessPass
                     + WEIGHT_HUE_HIST * (int)data.hueHistPass;
}
