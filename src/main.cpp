#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <vector>
#include <string>
#include <algorithm>

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    vector<string> fileList;

    DIR *d;
    struct dirent *dir;

    string dirName;

    int i = 0;

    if ((d = opendir(argv[1])) != NULL) {
        while ((dir = readdir(d)) != NULL) {
            dirName = dir->d_name;

            if (dirName.size() > 0 && dirName != "." && dirName != "..") {
                fileList.push_back(dirName);
            }
        }
        closedir (d);
    } else {
        cout << "Error opening directory";
        return EXIT_FAILURE;
    }


    Mat3b imageBgr, hsv;
    Mat1b mask1, mask2;
    Mat gray;

    vector<int> amountOfPixels;
    vector<int> areaOfHoles;
    vector<int> amountOfHoles;
    vector<Point> nonBlackPixels;


    // Implement filtering of images by amount of red colour in the photo

    for(int i = 0; i < fileList.size(); i++){
        Mat readImage;
        string separator = "/";
        string path = argv[1] + separator + fileList[i];

        imageBgr = imread(path);
        cvtColor(imageBgr, hsv, cv::COLOR_BGR2HSV);
        inRange(hsv, Scalar(0, 70, 50), Scalar(10, 255, 255), mask1);
        inRange(hsv, Scalar(120, 70, 50), Scalar(210, 255, 255), mask2);
        Mat1b mask = mask1 | mask2;

        findNonZero(mask, nonBlackPixels);
        amountOfPixels.push_back(nonBlackPixels.size());
    }


    int smallestArea = *min_element(amountOfPixels.begin(), amountOfPixels.end());
    int largestArea = *max_element(amountOfPixels.begin(), amountOfPixels.end());

    // Implement filtering of images by area of contours

    for(int i = 0; i < fileList.size(); i++) {
        Mat readImage;
        string separator = "/";
        string path = argv[1] + separator + fileList[i];

        imageBgr = imread(path);

        GaussianBlur( imageBgr, imageBgr, Size( 3, 3 ), 0, 0 );
        cvtColor(imageBgr, gray, cv::COLOR_BGR2GRAY);

        erode(gray, gray, Mat(), Point(-1, -1), 2, 1, 1);
        dilate(gray, gray, Mat(), Point(-1, -1), 2, 1, 1);
        cv::Canny(gray, gray, 200, 20);

        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;

        findContours( gray, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

         int contoursAmount = 0;
         areaOfHoles.clear();

         for(unsigned int i=0;i<contours.size();i++) {
            areaOfHoles.push_back(contourArea(contours[i]));
         }

        cout << areaOfHoles.size() << endl;
        amountOfHoles.push_back(areaOfHoles.size());

    }

    int smallestNumOfHoles = *min_element(amountOfHoles.begin(), amountOfHoles.end());
    int largestNumOfHoles = *max_element(amountOfHoles.begin(), amountOfHoles.end());

    int matchingPhotosCount = 0;

    for(int i = 0; i < amountOfHoles.size(); i++) {
        if ((amountOfHoles[i] >= 9) && (amountOfHoles[i] <= 306) && (amountOfPixels[i] >= 85) && (amountOfPixels[i] <= 3562)) {
            matchingPhotosCount++;
        }
    }


    cout << "Smallest area: " << smallestArea << endl;
    cout << "Largest area: " << largestArea << endl;
    cout << "Smallest number of holes: " << smallestNumOfHoles << endl;
    cout << "Largest number of holes: " << largestNumOfHoles << endl;
    cout << "Photos matching: " << matchingPhotosCount << "/" << fileList.size() << endl;

    return 0;
}
