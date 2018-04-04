/**
 * Benchmark various OpenCV operations
 *
 * Not particularly rigorous but just getting an idea of the cost of the various operations
 * used in this project.
 *
 * Dave Snowdon, 2017
 */

#include <stdlib.h>
#include <cstring>
#include <opencv2/opencv.hpp>

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>

int const TEST_ITERATIONS = 10000;
int const TEST_IMAGE_WIDTH = 500;

int const THRESHOLD_MIN = 127;
int const THRESHOLD_MAX = 255;

int const MOTION_BLUR_KERNEL_SIZE = 21;
int const MOTION_DILATE_KERNEL_SIZE = 3;
cv::Mat const MOTION_DILATE_STRUCTURING = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                                                    cv::Size(MOTION_DILATE_KERNEL_SIZE,
                                                                             MOTION_DILATE_KERNEL_SIZE));
int const MOTION_ERODE_KERNEL_SIZE = 2;
cv::Mat const MOTION_ERODE_STRUCTURING = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(MOTION_ERODE_KERNEL_SIZE,
                                                                                            MOTION_ERODE_KERNEL_SIZE));
cv::Point const MOTION_DILATE_ANCHOR = cv::Point(-1, -1);
int const MOTION_DILATE_ITERATIONS = 2;
double const MOTION_ACCUMULATOR_WEIGHT = 0.5;

dlib::frontal_face_detector face_detector = dlib::get_frontal_face_detector();

cv::Mat example_image;

cv::Mat example_small_image;

cv::Mat example_greyscale;

cv::Mat example_small_greyscale;

cv::Mat example_binary;

cv::Mat example_small_binary;

cv::Mat accumulator;

cv::Mat accumulator_small;

dlib::cv_image<dlib::bgr_pixel> example_dlib;

dlib::cv_image<dlib::bgr_pixel> example_small_dlib;

cv::Mat result_image;

// check cost of call via function pointer
void no_op() {
}

void resize_image() {
    double ratio = TEST_IMAGE_WIDTH / (double) example_image.cols;
    int height = (int) std::round(example_image.rows * ratio);
    cv::resize(example_image, result_image, cv::Size(TEST_IMAGE_WIDTH, height), 0, 0, cv::INTER_AREA);
}

void resize_then_greyscale() {
    double ratio = TEST_IMAGE_WIDTH / (double) example_image.cols;
    int height = (int) std::round(example_image.rows * ratio);
    cv::resize(example_image, result_image, cv::Size(TEST_IMAGE_WIDTH, height), 0, 0, cv::INTER_AREA);

    cv::cvtColor(result_image, result_image, cv::COLOR_BGR2GRAY);
}

void greyscale_then_resize() {
    cv::cvtColor(example_image, result_image, cv::COLOR_BGR2GRAY);

    double ratio = TEST_IMAGE_WIDTH / (double) example_image.cols;
    int height = (int) std::round(example_image.rows * ratio);
    cv::resize(result_image, result_image, cv::Size(TEST_IMAGE_WIDTH, height), 0, 0, cv::INTER_AREA);
}

void blur_large() {
    cv::GaussianBlur(example_image, result_image, cv::Size(MOTION_BLUR_KERNEL_SIZE, MOTION_BLUR_KERNEL_SIZE), 0);
}

void blur_small() {
    cv::GaussianBlur(example_small_image, result_image, cv::Size(MOTION_BLUR_KERNEL_SIZE, MOTION_BLUR_KERNEL_SIZE), 0);
}

void frame_difference_large() {
    cv::absdiff(example_image, example_image, result_image);
}

void frame_difference_small() {
    cv::absdiff(example_small_image, example_small_image, result_image);
}

void threshold_large() {
    cv::threshold(example_greyscale, result_image, THRESHOLD_MIN, THRESHOLD_MAX, cv::THRESH_BINARY);
}

void threshold_small() {
    cv::threshold(example_small_greyscale, result_image, THRESHOLD_MIN, THRESHOLD_MAX, cv::THRESH_BINARY);
}

void dilate_large() {
    cv::dilate(example_binary, result_image, MOTION_DILATE_STRUCTURING, MOTION_DILATE_ANCHOR, MOTION_DILATE_ITERATIONS);
}

void dilate_small() {
    cv::dilate(example_small_binary, result_image, MOTION_DILATE_STRUCTURING, MOTION_DILATE_ANCHOR,
               MOTION_DILATE_ITERATIONS);
}

void erode_large() {
    erode(example_binary, result_image, MOTION_ERODE_STRUCTURING);
}

void erode_small() {
    erode(example_small_binary, result_image, MOTION_ERODE_STRUCTURING);
}

void find_contours_large() {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(example_binary, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
}

void find_contours_small() {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(example_small_binary, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE,
                     cv::Point(0, 0));
}

void norm2_large() {
    double mean = cv::norm(example_binary, example_greyscale, cv::NORM_L2);
}

void norm2_small() {
    double mean = cv::norm(example_small_binary, example_small_greyscale, cv::NORM_L2);
}

void convert_to_float_large() {
    example_greyscale.convertTo(result_image, CV_32FC1);
}

void convert_to_float_small() {
    example_small_greyscale.convertTo(result_image, CV_32FC1);
}

void accumulate_weighted_large() {
    cv::accumulateWeighted(example_greyscale, accumulator, MOTION_ACCUMULATOR_WEIGHT);
}

void accumulate_weighted_small() {
    cv::accumulateWeighted(example_small_greyscale, accumulator_small, MOTION_ACCUMULATOR_WEIGHT);
}

void bitwise_and_large() {
    cv::bitwise_and(example_binary, example_binary, result_image);
}

void bitwise_and_small() {
    cv::bitwise_and(example_small_binary, example_small_binary, result_image);
}

void sum_large() {
    cv::Scalar sum = cv::sum(example_binary);
}

void sum_small() {
    cv::Scalar sum = cv::sum(example_small_binary);
}

void convert_dlib_large() {
    dlib::cv_image<dlib::bgr_pixel> converted(example_image);
}

void convert_dlib_small() {
    dlib::cv_image<dlib::bgr_pixel> converted(example_small_image);
}

void detect_faces_large() {
    std::vector<dlib::rectangle> faceRects = face_detector(example_dlib);
}

void detect_faces_small() {
    std::vector<dlib::rectangle> faceRects = face_detector(example_small_dlib);
}

/*
 * Time an operation specified via a function pointer. 
 * We assume that that the time taken to call the function whilst non-zero is small enough to 
 * not greatly affect the timings.
 */
void timer(void(*operation)(), const char *title) {
    std::cout << "Start: " << title << std::endl;
    double start_time = (double) cv::getTickCount();
    for (int i = 0; i < TEST_ITERATIONS; ++i) {
        operation();
    }
    double end_time = (double) cv::getTickCount();
    double operation_time = (end_time - start_time) / (cv::getTickFrequency() * TEST_ITERATIONS);
    std::cout << "End: " << title << " : " << operation_time << " seconds" << std::endl;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cout << "Usage: <filename>" << std::endl;
        return EXIT_FAILURE;
    }

    char *example_frame = argv[1];
    std::cout << "using " << example_frame << " as test image" << std::endl;

    /*
     * Setup test data
     */
    example_image = cv::imread(example_frame);

    double ratio = TEST_IMAGE_WIDTH / (double) example_image.cols;
    int height = (int) std::round(example_image.rows * ratio);
    cv::resize(example_image, example_small_image, cv::Size(TEST_IMAGE_WIDTH, height), 0, 0, cv::INTER_AREA);

    cv::cvtColor(example_image, example_greyscale, cv::COLOR_BGR2GRAY);
    cv::cvtColor(example_small_image, example_small_greyscale, cv::COLOR_BGR2GRAY);

    cv::threshold(example_greyscale, example_binary, 127, 255, cv::THRESH_BINARY);
    cv::threshold(example_small_greyscale, example_small_binary, 127, 255, cv::THRESH_BINARY);

    example_greyscale.convertTo(accumulator, CV_32FC1);
    example_small_greyscale.convertTo(accumulator_small, CV_32FC1);

    dlib::cv_image<dlib::bgr_pixel> example_dlib(example_image);
    dlib::cv_image<dlib::bgr_pixel> example_small_dlib(example_small_image);

    std::cout << "Size " << example_image.cols << "x" << example_image.rows << std::endl;
    std::cout << "Small size " << example_small_image.cols << "x" << example_small_image.rows << std::endl;
    std::cout << "Testing with " << TEST_ITERATIONS << " iterations" << std::endl;


    /*
     * Run benchmarks
     */
    timer(no_op, "Empty function");
    timer(resize_image, "Resize image");
    timer(resize_then_greyscale, "Resize then greyscale image");
    timer(greyscale_then_resize, "Greyscale then resize image");
    timer(blur_large, "Blur image (large)");
    timer(blur_small, "Blur image (small)");
    timer(frame_difference_large, "Frame difference (large)");
    timer(frame_difference_small, "Frame difference (small)");
    timer(threshold_large, "Threshold (large)");
    timer(threshold_small, "Threshold (small)");
    timer(dilate_large, "Dilate (large)");
    timer(dilate_small, "Dilate (small)");
    timer(erode_large, "Erode (large)");
    timer(erode_small, "Erode (small)");
    timer(find_contours_large, "Find contours (large)");
    timer(find_contours_small, "Find contours (small)");
    timer(norm2_large, "Norm2 (large)");
    timer(norm2_small, "Norm2 (small)");
    timer(convert_to_float_large, "Convert to float (large)");
    timer(convert_to_float_small, "Convert to float (small)");
    timer(accumulate_weighted_large, "Accumulate (large)");
    timer(accumulate_weighted_small, "Accumulate (small)");
    timer(bitwise_and_large, "Bitwise and (large)");
    timer(bitwise_and_small, "Bitwise and (small)");
    timer(sum_large, "Sum (large)");
    timer(sum_small, "Sum (small)");
    timer(convert_dlib_large, "Convert image to dlib (large)");
    timer(convert_dlib_small, "Convert image to dlib (small)");
    timer(detect_faces_large, "Detect faces (large)");
    timer(detect_faces_small, "Detect faces (small)");
}
