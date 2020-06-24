
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
extern "C" void testcall_cpp(float value)
{
    Mat image;
    image = imread( "/Users/yongyutjantaboot/CLionProjects/rust-and-cmake88/4.jpg", 1 );

    if ( !image.data )
    {
        printf("No image data \n");

    }
    imshow("ssss",image);
    waitKey(0);
    std::cout << " ooooooo Hello, world from C++! Value passed: " << value << std::endl;
}
