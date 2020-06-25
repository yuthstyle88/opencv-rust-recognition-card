
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
extern "C" {
    int testcall_cpp(Mat image) {
        imshow("ssss", image);
        waitKey(0);
        // std::cout << " ooooooo Hello, world from C++! Value passed: " << value << std::endl;
        return (16);
    }

    void ex_display_imag_and_wait(Mat mat) {
        imshow("Display image", mat);
        waitKey(0);
    }
}