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
#include <opencv2/objdetect.hpp>

#include <dlib/opencv.h>
#include <dlib/dnn.h>
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

dlib::shape_predictor landmark_detector;

// From dnn_face_recognition_ex.cpp
// ----------------------------------------------------------------------------------------

// The next bit of code defines a ResNet network.  It's basically copied
// and pasted from the dnn_imagenet_ex.cpp example, except we replaced the loss
// layer with loss_metric and made the network somewhat smaller.  Go read the introductory
// dlib DNN examples to learn what all this stuff means.
//
// Also, the dnn_metric_learning_on_images_ex.cpp example shows how to train this network.
// The dlib_face_recognition_resnet_model_v1 model used by this example was trained using
// essentially the code shown in dnn_metric_learning_on_images_ex.cpp except the
// mini-batches were made larger (35x15 instead of 5x5), the iterations without progress
// was set to 10000, and the training dataset consisted of about 3 million images instead of
// 55.  Also, the input layer was locked to images of size 150.
template<template<int, template<typename> class, int, typename> class block, int N,
        template<typename> class BN, typename SUBNET>
using residual = dlib::add_prev1<block<N, BN, 1, dlib::tag1<SUBNET>>>;

template<template<int, template<typename> class, int, typename> class block, int N,
        template<typename> class BN, typename SUBNET>
using residual_down = dlib::add_prev2<dlib::avg_pool<2, 2, 2, 2, dlib::skip1<dlib::tag2<block<N, BN, 2, dlib::tag1<SUBNET>>>>>>;

template<int N, template<typename> class BN, int stride, typename SUBNET>
using block  = BN<dlib::con<N, 3, 3, 1, 1, dlib::relu<BN<dlib::con<N, 3, 3, stride, stride, SUBNET>>>>>;

template<int N, typename SUBNET> using ares      = dlib::relu<residual<block, N, dlib::affine, SUBNET>>;
template<int N, typename SUBNET> using ares_down = dlib::relu<residual_down<block, N, dlib::affine, SUBNET>>;

template<typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
template<typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template<typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template<typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template<typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = dlib::loss_metric<dlib::fc_no_bias<128, dlib::avg_pool_everything<
        alevel0<alevel1<alevel2<alevel3<alevel4<dlib::max_pool<3, 3, 2, 2, dlib::relu<dlib::affine<dlib::con<32, 7, 7, 2, 2,
                dlib::input_rgb_image_sized<150>
        >>>>>>>>>>>>;

// DNN used for face recognition
anet_type face_metrics_net;

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

dlib::rectangle face_bounds_large;

dlib::rectangle face_bounds_small;

dlib::full_object_detection landmarks_large;

dlib::full_object_detection landmarks_small;

dlib::matrix<float, 0, 1> face_descriptor_result;

dlib::full_object_detection landmarks_result;

dlib::matrix<dlib::rgb_pixel> face_chip_result;

dlib::matrix<dlib::rgb_pixel> face_image;

std::vector<dlib::matrix<dlib::rgb_pixel>> face_images;

dlib::correlation_tracker tracker_large;

dlib::correlation_tracker tracker_small;

double tracker_confidence_result;

cv::CascadeClassifier faceCascade;

std::vector<cv::Rect> faces_result;

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

void detect_faces_opencv_large() {
    /*
     * Using a global vector here to try and prevent the compile from optimising away the call.
     * Expectation that cost of std::vector.clear() should be small relative to the cost of face detection.
     */
    faces_result.clear();
    faceCascade.detectMultiScale(example_greyscale, faces_result, 1.2, 2);
}

void detect_faces_opencv_small() {
    /*
     * Using a global vector here to try and prevent the compile from optimising away the call.
     * Expectation that cost of std::vector.clear() should be small relative to the cost of face detection.
     */
    faces_result.clear();
    faceCascade.detectMultiScale(example_small_greyscale, faces_result, 1.2, 2);
}

void face_landmarks_large() {
    landmarks_result = landmark_detector(example_dlib, face_bounds_large);
}

void face_landmarks_small() {
    landmarks_result = landmark_detector(example_small_dlib, face_bounds_small);
}

void extract_face_chip_large() {
    dlib::extract_image_chip(example_dlib, dlib::get_face_chip_details(landmarks_large, 150, 0.25), face_chip_result);
}

void extract_face_chip_small() {
    dlib::extract_image_chip(example_small_dlib, dlib::get_face_chip_details(landmarks_small, 150, 0.25), face_chip_result);
}

// face descriptors are always computed on the same size image
void compute_face_descriptor() {
    auto descriptors = face_metrics_net(face_images);
    face_descriptor_result = descriptors[0];
}

void correlation_tracker_update_large() {
    tracker_confidence_result = tracker_large.update(example_dlib);
}

void correlation_tracker_update_small() {
    tracker_confidence_result = tracker_small.update(example_small_dlib);
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

    // facial landmark detector
    dlib::deserialize("models/shape_predictor_5_face_landmarks.dat") >> landmark_detector;

    // DNN used for face recognition
    dlib::deserialize("models/dlib_face_recognition_resnet_model_v1.dat") >> face_metrics_net;

    if (!faceCascade.load("/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml")) {
        std::cerr << "Error loading face cascade" << std::endl;
        return 1;
    }

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

    dlib::cv_image<dlib::bgr_pixel> tmp_dlib(example_image);
    example_dlib = tmp_dlib;
    dlib::cv_image<dlib::bgr_pixel> tmp_small_dlib(example_small_image);
    example_small_dlib = tmp_small_dlib;

    std::vector<dlib::rectangle> faceRectsLarge = face_detector(example_dlib);
    if (0 == faceRectsLarge.size()) {
        std::cerr << "Example image must contain at least one face" << std::endl;
        return 1;
    }
    face_bounds_large = faceRectsLarge[0];

    /*
     * dlib needs faces to be about 80x80 pixels in order to detect them
     */
    bool do_small_face_tests = true;
    std::vector<dlib::rectangle> faceRectsSmall = face_detector(example_small_dlib);
    if (0 == faceRectsSmall.size()) {
        std::cerr << "Can't find face in small images, skipping tests" << std::endl;
        do_small_face_tests = false;
    } else {
        face_bounds_small = faceRectsSmall[0];
    }

    landmarks_large = landmark_detector(example_dlib, face_bounds_large);

    dlib::extract_image_chip(example_dlib, dlib::get_face_chip_details(landmarks_large, 150, 0.25), face_image);
    face_images.push_back(face_image);

    dlib::rectangle padded_rectangle_large(face_bounds_large.left() - 10,
                                     face_bounds_large.top() - 20,
                                     face_bounds_large.right() + 10,
                                     face_bounds_large.bottom() + 20);
    tracker_large.start_track(example_dlib, padded_rectangle_large);

    if (do_small_face_tests) {
        landmarks_small = landmark_detector(example_small_dlib, face_bounds_small);

        dlib::rectangle padded_rectangle_small(face_bounds_small.left() - 10,
                                               face_bounds_small.top() - 20,
                                               face_bounds_small.right() + 10,
                                               face_bounds_small.bottom() + 20);
        tracker_small.start_track(example_small_dlib, padded_rectangle_small);
    }

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

    timer(face_landmarks_large, "Face landmarks (large)");
    if (do_small_face_tests) {
        timer(face_landmarks_small, "Face landmarks (small)");
    }
    timer(extract_face_chip_large, "Extract face chip (large)");
    if (do_small_face_tests) {
        timer(extract_face_chip_small, "Extract face chip (small)");
    }
    timer(compute_face_descriptor, "Face descriptor");


    /*
     * These only time a single frame update so they are not great overall tests of tracker
     * performance.
     */
    timer(correlation_tracker_update_large, "dlib correlation tracker update (large)");
    if (do_small_face_tests) {
        timer(correlation_tracker_update_small, "dlib correlation tracker update (small)");
    }

    // Slow operations
    timer(detect_faces_large, "dlib detect faces (large)");
    timer(detect_faces_small, "dlib detect faces (small)");
    timer(detect_faces_opencv_large, "OpenCV detect faces (large)");
    timer(detect_faces_opencv_small, "OpenCV detect faces (small)");

    return 0;
}
