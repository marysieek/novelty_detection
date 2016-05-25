# include <iostream>
# include <string>
# include <vector>
# include <sstream>

# include "opencv2/opencv_modules.hpp"
# include "opencv2/core/core.hpp"
# include "opencv2/features2d/features2d.hpp"
# include "opencv2/highgui/highgui.hpp"
# include "opencv2/imgproc/imgproc.hpp"
# include "opencv2/nonfree/features2d.hpp"

using namespace std;
using namespace cv;

// acceptable value of matches, determined by trial and error
const int ACCEPTABLE_MATCH = 52;

// determine min hessian value
const int MIN_HESSIAN = 1000;

// distance coefficient, determined by trial and error
const double DIST_COEFF = 0.7;

// max and min values of non black pixels, determined by trial and error
const int NON_BLACK_MAX = 3562;
const int NON_BLACK_MIN = 85;

int main( int argc, char** argv )
{
  // check arguments
  if (argc < 3) {
    cerr << "You should give 2 directories" << endl;
    exit (-1);
  }

  // load files from training and (testing or novelty) directory
  vector<string> fileNamesInDir1, fileNamesInDir2;

  glob((string(argv[1]) + "/*.jpg"), fileNamesInDir1, false);
  glob((string(argv[2]) + "/*.jpg"), fileNamesInDir2, false);

  auto numberOfFilesInDir1 = fileNamesInDir1.size();
  auto numberOfFilesInDir2 = fileNamesInDir2.size();

  cout << "Found " << numberOfFilesInDir1 << " files in " << argv[1] << endl;
  cout << "Found " << numberOfFilesInDir2 << " files in " << argv[2] << endl;

  // detect the keypoints using SURF Detector
  SurfFeatureDetector detector(MIN_HESSIAN);

  vector<KeyPoint> keypoints_1, keypoints_2;

  // initialize counter for photos
  int matchingPhotosCount = 0;

  // number of pixels that are not black in photo
  vector<Point> nonBlackPixels;

  // count matches after using FLANN method
  vector<int> matchesCount;

  Mat descriptors_1, descriptors_2;

  // calculate descriptors (feature vectors)
  SurfDescriptorExtractor extractor;
  FileStorage store("template.xml", FileStorage::WRITE);

  int counter = 0;

  // iterate over images found in training directory
  for (auto file : fileNamesInDir1) {
    ostringstream outputName;

    counter++;
    Mat image = imread(file.c_str(), CV_LOAD_IMAGE_GRAYSCALE);

    detector.detect(image, keypoints_1);
    extractor.compute(image, keypoints_1, descriptors_1);

    outputName << "descriptors_" << counter;

    write(store, outputName.str(), descriptors_1);
  }

  cout << "Analysis finished" << endl;

  store.release();

  // open storage file
  store.open("template.xml", FileStorage::READ);

  // iterate over images found in testing or novelty directory
  for (auto testFile : fileNamesInDir2) {
    matchesCount.clear();

    cout << endl << "Analyzing file: " << testFile << endl;
    Mat testImage = imread(testFile.c_str(), CV_LOAD_IMAGE_GRAYSCALE);

    detector.detect(testImage, keypoints_1);
    extractor.compute(testImage, keypoints_1, descriptors_1);

    // compare against every image from dir1
    for (size_t i=1; i<= fileNamesInDir1.size(); i++) {
      ostringstream inputName;
      inputName << "descriptors_" << i;
      store[inputName.str()] >>  descriptors_2;

      cout << "Matching against " << inputName.str() << endl;
      // match descriptor vectors using FLANN matcher
      FlannBasedMatcher matcher;
      vector< vector<DMatch> > matches;

      cout << "DESCRIPTOR 1 ROWS " << descriptors_1.rows << endl;
      cout << "DESCRIPTOR 2 ROWS " << descriptors_2.rows << endl;
      matcher.knnMatch(descriptors_1, descriptors_2, matches, 2);

      // max and min distance between keypoints
      int max_dist = 0;
      int min_dist = 100;

      // calculate max and min distances between keypoints
      for(int i = 0; i < descriptors_1.rows; i++) {
        double dist = matches[i][0].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
      }

      vector<DMatch> goodMatches;

      for(int i = 0; i < descriptors_1.rows; i++) {
        const DMatch &m1 = matches[i][0];
        const DMatch &m2 = matches[i][1];

        if (m1.distance <= DIST_COEFF * m2.distance) {
          goodMatches.push_back(m1);
        }
      }

      cout << goodMatches.size() << endl;
      matchesCount.push_back(goodMatches.size());
    }

    // display all matches
    cout << "[";
    for (int i = 0; i < (int)matchesCount.size(); i++) {
      cout << matchesCount[i];
      if (i < (int)matchesCount.size() - 1){
        cout << ", ";
      }
    }
    cout << "]" << endl;

    // calculate smallest and largest number of matches
    int smallestNumOfMatches = *min_element(matchesCount.begin(), matchesCount.end());
    int largestNumOfMatches = *max_element(matchesCount.begin(), matchesCount.end());


    // display smallest and largest number of matches
    cout << "Smallest number of matches: " << smallestNumOfMatches << endl;
    cout << "Largest number of matches: " << largestNumOfMatches << endl;

    // implement matching by red colour
    Mat1b mask1, mask2;
    Mat3b hsv;

    Mat3b imageBgr = imread(testFile.c_str());
    cvtColor(imageBgr, hsv, COLOR_BGR2HSV);

    // determine range of redness
    inRange(hsv, Scalar(0, 70, 50), Scalar(10, 255, 255), mask1);
    inRange(hsv, Scalar(120, 70, 50), Scalar(210, 255, 255), mask2);

    Mat1b mask = mask1 | mask2;

    // find nonzero pixels
    findNonZero(mask, nonBlackPixels);

    cout << "Amount of nonBlackPixels: " << nonBlackPixels.size() << endl;

    if (largestNumOfMatches >= ACCEPTABLE_MATCH &&
      (nonBlackPixels.size() >= NON_BLACK_MIN) &&
      (nonBlackPixels.size() <= NON_BLACK_MAX)) {

      matchingPhotosCount++;
    }
  }

  store.release();

  cout << "Total photos matching: " << matchingPhotosCount << "/" << fileNamesInDir2.size() << endl;

  waitKey(0);

  return 0;
}

