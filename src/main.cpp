#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>

using namespace cv;
using namespace std;

int main() {
        Mat OurImage;
        string Destination = "Lena.jpg";
        OurImage = imread(Destination, CV_LOAD_IMAGE_GRAYSCALE);
        if (!OurImage.data) {
                printf("No image!");
                getchar();
                return -1;
        }
        namedWindow("WINDOW", CV_WINDOW_AUTOSIZE);
        imshow("WINDOW", OurImage);
        waitKey(0);
}
