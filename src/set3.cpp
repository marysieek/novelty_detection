
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
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
const int HUE_HIST_MIN_MATCHES = 6;
const float WEIGHT_ROUNDNESS = 0.2;
const float WEIGHT_HUE_HIST = 0.8;
const int OBJECT_TYPES = 6;

typedef struct {
        string fileName;
        double roundness;
        bool roundnessPass;
        vector<double> hu;
        Mat hueImage;
        Mat hueHist;
        bool hueHistPass;
        float score;
        vector<int> huLabels;
} ObjectData;

void sortFiles (vector<string> &names);
vector<ObjectData> analyzeImages (vector<string> &fileNames);
void compareRoundness(vector<ObjectData> training, vector<ObjectData> testing);
void compareHueHistograms(vector<ObjectData> training, vector<ObjectData> testing);
void prepareImageMats(Mat &colorImage, Mat &grayImage, Mat &contourImage);
void cleanContoursWithSigma(vector<Point> &points, double maxDistanceSigma);
void printData(ObjectData &data);
void calculateScore(ObjectData &data);
void clusterHuMoments(vector<ObjectData> &data);

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
        // auto numberOfFiles = trainingFileNames.size();

        // sortFiles(trainingFileNames);
        sortFiles(testingFileNames);

        // cout << "Found " << numberOfFiles << " files in " << argv[1] << endl;

        namedWindow("COLOR");
        namedWindow("GRAY");
        namedWindow("CONTOUR");
        namedWindow("HUE");

        auto trainingData = analyzeImages(trainingFileNames);
        auto testingData = analyzeImages(testingFileNames);
        // clusterHuMoments(trainingData);
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
                // cout << endl << "Analyzing file: " << file << endl;
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
                vector<Point> hull;
                convexHull(contourPoints, hull);
                auto area = contourArea(hull);

                Point2f circleCenter;
                float circleRadius;
                minEnclosingCircle(hull, circleCenter, circleRadius);
                data.roundness = sqrt(area / (M_PI * pow(circleRadius, 2)));

                // calculate contour moments

                Moments mu = moments(contourPoints, true);
                HuMoments(mu, data.hu);

                // calculate hue histograms
                Mat hsvImage;
                Mat hsvPlanes[3];

                cvtColor(colorImage, hsvImage, CV_BGR2HSV);
                split(hsvImage, hsvPlanes);
                data.hueImage = hsvPlanes[0];

                int hueHistSize = HUE_HIST_BINS;
                float range [] = {0, 180};
                const float *hueHistRange = range;
                calcHist(&data.hueImage, 1, 0, Mat(), data.hueHist, 1, &hueHistSize, &hueHistRange, 1, 0);
                normalize(data.hueHist, data.hueHist, 0, 1, NORM_MINMAX, -1, Mat());

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

void sortFiles (vector<string> &names){
        sort (names.begin(), names.end(), [](std::string s1, std::string s2) {
                // cout << s1.substr(s1.find_last_of("/") + 1, s1.find_last_of(".")) << endl;
                auto number1 = std::stoi(s1.substr(s1.find_last_of("/") + 1, s1.find_last_of(".")));
                auto number2 = std::stoi(s2.substr(s2.find_last_of("/") + 1, s2.find_last_of(".")));
                return number1 < number2;
        });
}

// void clusterHuMoments(vector<ObjectData> &data) {
//         const int m = 1;
//         vector<Point2f> hu0Samples(data.size());
//         transform(
//                 data.begin(),
//                 data.end(),
//                 hu0Samples.begin(),
//                 [m](ObjectData &object) {
//                 return Point2f(object.hu[m], 0);
//         });
//         double eps = accumulate(hu0Samples.begin(), hu0Samples.end(), Point2f(0.0, 0.0)).x / hu0Samples.size() * 0.000001;
//         cout << "K-means epsilon: " << eps << endl;
//         vector<int> huLabels;
//         kmeans(hu0Samples, OBJECT_TYPES, huLabels, TermCriteria(2, 1000, eps), 1, KMEANS_RANDOM_CENTERS);
//         cout << "Samples: " << hu0Samples.size() << endl;
//         cout << "huLabels: " << huLabels.size() << endl;
//         cout << "data: " << data.size() << endl;
//         for(unsigned i = 0; i < data.size(); i++) {
//                 data[i].huLabels.push_back(huLabels[i]);
//                 cout << huLabels[i];
//         }
//         cout << endl;
// }

void compareRoundness(vector<ObjectData> training, vector<ObjectData> testing) {
        // cout << "Roundness tests" << endl;
        double minRoundness = min_element(
                training.begin(),
                training.end(),
                [](ObjectData o1, ObjectData o2) {
                return o1.roundness < o2.roundness;
        })->roundness;
        unsigned passed = 0;
        unsigned failed = 0;

        // cout << "\tminimal roundness found in training: " << minRoundness << endl;
        minRoundness *= 1.0 - ROUNDNESS_LIMIT_MARGIN;
        // cout << "\troundness lower limit for pass: " << minRoundness << endl;

        for (auto object : testing) {
                if (object.roundness > minRoundness) {
                        passed++;
                        object.roundnessPass = true;
                } else {
                        failed++;
                        object.roundnessPass = false;
                }
        }

        // cout << "passed : " << passed << endl;
        // cout << "failed : " << failed << endl;

}

void compareHueHistograms(vector<ObjectData> training, vector<ObjectData> testing) {
        // cout << "Hue histogram tests" << endl;
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
                // cout << "File: " << testObject.fileName << endl;
                // cout << "\tMatches : " << matches << endl;

                if (matches >= HUE_HIST_MIN_MATCHES) {
                        passed++;
                        testObject.hueHistPass = true;
                } else {
                        failed++;
                        testObject.hueHistPass = false;
                }

                auto fileNameOnly = testObject.fileName.substr(testObject.fileName.find_last_of('/')+1);
                cout << fileNameOnly << "\t" << (int)(!testObject.hueHistPass) << endl;
        }

        // cout << "passed : " << passed << endl;
        // cout << "failed : " << failed << endl;
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
        auto sum = std::accumulate(points.begin(), points.end(), Point(0, 0));
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
                sqrt(std::accumulate(pointDistances.begin(), pointDistances.end(), 0.0,
                                     [](double x, double y) {
                return x + y * y;
        }) / pointDistances.size());

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
