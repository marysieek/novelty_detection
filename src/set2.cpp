#include <iostream>
#include <string>
#include <vector>

# include "opencv2/opencv_modules.hpp"
# include "opencv2/core/core.hpp"
# include "opencv2/features2d/features2d.hpp"
# include "opencv2/highgui/highgui.hpp"
# include "opencv2/nonfree/features2d.hpp"

using namespace std;
using namespace cv;

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
  int minHessian = 400;

  SurfFeatureDetector detector(minHessian);

  vector<KeyPoint> keypoints_1, keypoints_2;

  // iterate over images found in testing or novelty directory

  for (auto file : fileNamesInDir2) {
    cout << endl << "Analyzing file: " << file << endl;
    Mat image = imread(file.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
    vector<int> matchesCount;

    matchesCount.clear();

    // iterate over images found in training directory

    for (auto testFile : fileNamesInDir1) {

      Mat testImage = imread(testFile.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
      detector.detect(image, keypoints_1);
      detector.detect(testImage, keypoints_2);

      // calculate descriptors (feature vectors)
      SurfDescriptorExtractor extractor;

      Mat descriptors_1, descriptors_2;

      extractor.compute(image, keypoints_1, descriptors_1);
      extractor.compute(testImage, keypoints_2, descriptors_2);

      // match descriptor vectors using FLANN matcher
      FlannBasedMatcher matcher;
      vector< DMatch > matches;
      matcher.match(descriptors_1, descriptors_2, matches);

      double max_dist = 0; double min_dist = 100;

      // calculate max and min distances between keypoints
      for( int i = 0; i < descriptors_1.rows; i++ ) {
        double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
      }

      vector< DMatch > good_matches;

      for( int i = 0; i < descriptors_1.rows; i++ ) {
        if( matches[i].distance <= max(2*min_dist, 0.02) ) {
          good_matches.push_back(matches[i]);
        }
      }

      matchesCount.push_back(good_matches.size());
    }

    // calculat smallest and largest number of matches
    int smallestNumOfMatches = *min_element(matchesCount.begin(), matchesCount.end());
    int largestNumOfMatches = *max_element(matchesCount.begin(), matchesCount.end());

    // display all matches
    cout << "[";
    for (int i = 0; i < (int)matchesCount.size(); i++) {
      cout << matchesCount[i];
      if (i < (int)matchesCount.size() - 1){
        cout << ", ";
      }
    }
    cout << "]" << endl;

    // display smallest and largest number of matches
    cout << "Smallest number of matches: " << smallestNumOfMatches << endl;
    cout << "Largest number of matches: " << largestNumOfMatches << endl;
  }

  waitKey(0);

  return 0;
}
