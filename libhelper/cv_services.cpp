
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;
extern "C" {

int thresh = 70, N = 11;
const char* wndname = "Square Detection Demo";
int morph_elem = 0;
int morph_size = 1;
int morph_operator = 0;

static double angle( Point pt1, Point pt2, Point pt0 )
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}
// returns sequence of squares detected on the image.
static void findSquares( const Mat& image, vector<vector<Point> >& squares )
{
    squares.clear();
    Mat pyr, timg, gray0(image.size(), CV_8U), gray;
    // down-scale and upscale the image to filter out the noise
    pyrDown(image, pyr, Size(image.cols/2, image.rows/2));
    pyrUp(pyr, timg, image.size());
    vector<vector<Point> > contours;
    // find squares in every color plane of the image
    for( int c = 0; c < 3; c++ )
    {
        int ch[] = {c, 0};
        mixChannels(&timg, 1, &gray0, 1, ch, 1);
        // try several threshold levels
        for( int l = 0; l < N; l++ )
        {
            // hack: use Canny instead of zero threshold level.
            // Canny helps to catch squares with gradient shading
            if( l == 0 )
            {
                // apply Canny. Take the upper threshold from slider
                // and set the lower to 0 (which forces edges merging)
                Canny(gray0, gray, 0, thresh, 5);
                // dilate canny output to remove potential
                // holes between edge segments
                dilate(gray, gray, Mat(), Point(-1,-1));
            }
            else
            {
                // apply threshold if l!=0:
                //     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
                gray = gray0 >= (l+1)*255/N;
            }
            // find contours and store them all as a list
            findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
            vector<Point> approx;
            // test each contour
            for( size_t i = 0; i < contours.size(); i++ )
            {
                // approximate contour with accuracy proportional
                // to the contour perimeter
                approxPolyDP(contours[i], approx, arcLength(contours[i], true)*0.02, true);
                // square contours should have 4 vertices after approximation
                // relatively large area (to filter out noisy contours)
                // and be convex.
                // Note: absolute value of an area is used because
                // area may be positive or negative - in accordance with the
                // contour orientation
                if( approx.size() == 4 &&
                    fabs(contourArea(approx)) > 1000 &&
                    isContourConvex(approx) )
                {
                    double maxCosine = 0;
                    for( int j = 2; j < 5; j++ )
                    {
                        // find the maximum cosine of the angle between joint edges
                        double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
                        maxCosine = MAX(maxCosine, cosine);
                    }
                    // if cosines of all angles are small
                    // (all angles are ~90 degree) then write quandrange
                    // vertices to resultant sequence
                    if( maxCosine < 0.1)
                        squares.push_back(approx);
                }
            }
        }
    }
}
// the function draws all the squares in the image
static void drawSquares( Mat& image, const vector<vector<Point> >& squares )
{
    for( size_t i = 0; i < squares.size(); i++ )
    {
        const Point* p = &squares[i][0];
        int n = (int)squares[i].size();
        polylines(image, &p, &n, 1, true, Scalar(0,255,0), 3, LINE_AA);
    }
    imshow(wndname, image);
}

bool check_is_red(Mat *image){
    bool is_red = false;
    Mat3b hsv;
    cvtColor(*image, hsv, COLOR_BGR2HSV);
    //imshow("Before Mask Red", *image);
    Mat1b mask1, mask2;
    inRange(hsv, Scalar(0, 70, 50), Scalar(10, 255, 255), mask1);
    inRange(hsv, Scalar(170, 70, 50), Scalar(180, 255, 255), mask2);
    int match_red = 0;
    Mat1b mask = mask1 | mask2;

    for (int y = 0; y < mask.rows; y++) {
        for (int x = 0; x < mask.cols; x++) {
            char16_t at_color = mask.at<uchar>(y, x);
            //cout <<"at_color : "<< at_color <<endl;
            if (at_color == 255)
                match_red++;
        }
    }
    if (match_red > 10) is_red = true;

   // cout <<"match_red : "<< match_red <<endl;
   // waitKey();
    return is_red;
}

Mat* auto_close_line(Mat *image)
{
    // Since MORPH_X : 2,3,4,5 and 6
    Mat *dst = new Mat();
    Mat*  gray = new Mat();
   // imshow( "gray", image );
    int morph_elem = 0;
    int operation = 4;
    int morph_size = 1;
    cvtColor(*image, *gray, COLOR_BGR2GRAY);
    //image2 = gray;
    //Canny(*gray, *gray, 100, 190, 5);

    //int c = waitKey();
    //cvtColor(gray, gray, COLOR_BGR2GRAY);
    threshold(*gray, *image, 190, 255, THRESH_BINARY_INV);
    //imshow( "threshold", *image );
   // int c = waitKey();
    //fastNlMeansDenoising(gray, dst, 30.0, 7, 21);
    //imshow( "auto_close_line", dst );
    Mat element = getStructuringElement( morph_elem, Size( 2*morph_size , 2*morph_size), Point( morph_size, morph_size ) );

    /// Apply the specified morphology operation
    morphologyEx( *image, *dst, operation, element );
    //imshow( "auto_close_line", *dst );
    return dst;
}


/*

int has_square(Mat &image) {
    auto_close_line(image);
  */
/*  vector<vector<Point> > squares;
    findSquares(image, squares);
    drawSquares(image, squares);*//*


    int c = waitKey();
    if( c == 27 )

        return 0;
    // std::cout << " ooooooo Hello, world from C++! Value passed: " << value << std::endl;
    return (16);
}
*/
    Mat* test_image(Mat *image) {
        Mat *dst = new Mat();
        Mat *gray = new Mat();

        cvtColor(*image, *gray, COLOR_BGR2GRAY);
    //    threshold(gray, image, 190, 255, THRESH_BINARY_INV);
    //    int operation = 4;
    //    Mat element = getStructuringElement( morph_elem, Size( 2*morph_size , 2*morph_size), Point( morph_size, morph_size ) );
    //    morphologyEx( *image, *dst, operation, element );

        return gray;
    }
}