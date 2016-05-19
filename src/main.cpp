#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
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

    vector<int> amountOfPixels;
    vector<Point> nonBlackPixels;


    for(int i=0; i < fileList.size(); i++){
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
        cout << nonBlackPixels.size() << endl;

        //namedWindow( "window", WINDOW_AUTOSIZE );// Create a window for display.
        //imshow("window", mask);
        //waitKey(1000);
    }


    int smallestArea = *min_element(amountOfPixels.begin(), amountOfPixels.end());
    int largestArea = *max_element(amountOfPixels.begin(), amountOfPixels.end());
    int matchingPhotosCount = 0;

    for(int i = 0; i < amountOfPixels.size(); i++) {
        if ((amountOfPixels[i] >= 85) && (amountOfPixels[i] <= 3562)) {
            matchingPhotosCount++;
        }
    }

    cout << "Smallest area: " << smallestArea << endl;
    cout << "Largest area: " << largestArea << endl;
    cout << "Photos matching: " << matchingPhotosCount << "/" << fileList.size() << endl;

    return 0;
}
