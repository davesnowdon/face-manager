/*
 *  Face tracker 0.1
 *   Implementation of motion detector algorithms
 *
 *  Copyright (c) 2018 David Snowdon. All rights reserved.
 *
 *  Distributed under the Boost Software License, Version 1.0. (See accompanying
 *  file LICENSE.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include "motiondetector.h"

#include <stdlib.h>

int const MOTION_BLUR_KERNEL_SIZE = 21;
int const MOTION_THRESH_MIN = 25;
int const MOTION_THRESH_MAX = 255;
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

cv::Mat
resizeToWidth(cv::Mat src, int width) {
    cv::Mat dest;
    int rows = src.rows;
    int cols = src.cols;
    double ratio = width / (double) cols;
    int height = (int) std::round(rows * ratio);
    cv::resize(src, dest, cv::Size(width, height), 0, 0, cv::INTER_AREA);
    return dest;
}


void
ContourMotionDetector::initFrame(cv::Mat frame) {
    logger.debug("ContourMotionDetector::first-frame", frame);
    cv::Mat processed = preProcessImage(frame);
    logger.debug("ContourMotionDetector::first-frame-processed", processed);
    processed.convertTo(accumulator_, CV_32FC1);
}


bool
ContourMotionDetector::detectMotion(cv::Mat frame) {
    cv::Mat cur = preProcessImage(frame);
    logger.debug("ContourMotionDetector::pre-process", cur);

    // We need to conver the accumulator back to 8bit values for comparison
    cv::Mat abs_accumulator;
    cv::convertScaleAbs(accumulator_, abs_accumulator);
    logger.debug("ContourMotionDetector::abs_accumulator", abs_accumulator);

    // accumlate running averate of frames seen so far
    cv::accumulateWeighted(cur, accumulator_, MOTION_ACCUMULATOR_WEIGHT);
    logger.trace("ContourMotionDetector::accumulator", accumulator_);

    // difference between accumulator and current frame
    cv::Mat diff;
    cv::absdiff(cur, abs_accumulator, diff);
    logger.trace("ContourMotionDetector::diff", diff);

    // binarise
    cv::Mat thres;
    cv::threshold(diff, thres, MOTION_THRESH_MIN, MOTION_THRESH_MAX, cv::THRESH_BINARY);
    logger.trace("ContourMotionDetector::threshold", thres);

    cv::Mat dilated;
    cv::dilate(thres, dilated, MOTION_DILATE_STRUCTURING, MOTION_DILATE_ANCHOR, MOTION_DILATE_ITERATIONS);
    logger.debug("ContourMotionDetector::dilated", dilated);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(dilated, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

    // See if any contours are bigger than the threshold
    for (std::vector<std::vector<cv::Point>>::iterator it = contours.begin(); it != contours.end(); ++it) {
        double area = cv::contourArea(*it);
        if (area > motion_dectected_area_) {
            return true;
        }
    }

    return false;
}


cv::Mat
ContourMotionDetector::preProcessImage(cv::Mat frame) {
    cv::Mat small = resizeToWidth(frame, image_width_);
    cv::Mat grey;
    cv::cvtColor(small, grey, cv::COLOR_BGR2GRAY);
    cv::Mat blurred;
    cv::GaussianBlur(grey, blurred, cv::Size(MOTION_BLUR_KERNEL_SIZE, MOTION_BLUR_KERNEL_SIZE), 0);
    return blurred;
}


void
MeanSquaredErrorMotionDetector::initFrame(cv::Mat frame) {
    logger.debug("MeanSquaredErrorMotionDetector::first-frame", frame);
    accumulator_ = preProcessImage(frame);
    logger.debug("MeanSquaredErrorMotionDetector::first-frame-processed", accumulator_);
}


bool
MeanSquaredErrorMotionDetector::detectMotion(cv::Mat frame) {
    cv::Mat cur = preProcessImage(frame);
    logger.debug("MeanSquaredErrorMotionDetector::pre-process", cur);

    double mean = cv::norm(cur, accumulator_, cv::NORM_L2);
    //std::cout << "mean = " << mean << std::endl;

    // accumlate running averate of frames seen so far
    cv::accumulateWeighted(cur, accumulator_, MOTION_ACCUMULATOR_WEIGHT);
    logger.trace("MeanSquaredErrorMotionDetector::accumulator", accumulator_);

    return mean > threshold_;
}


cv::Mat
MeanSquaredErrorMotionDetector::preProcessImage(cv::Mat frame) {
    cv::Mat small = resizeToWidth(frame, image_width_);
    cv::Mat grey;
    cv::cvtColor(small, grey, cv::COLOR_BGR2GRAY);
    cv::Mat flt;
    grey.convertTo(flt, CV_32FC1);

    if (use_blur_) {
        cv::Mat blurred;
        cv::GaussianBlur(flt, blurred, cv::Size(MOTION_BLUR_KERNEL_SIZE, MOTION_BLUR_KERNEL_SIZE), 0);
        return blurred;
    } else {
        return flt;
    }
}


void
FrameDifferenceMotionDetector::initFrame(cv::Mat frame) {
    if (0 == prev_frame_.cols) {
        prev_frame_ = preProcessImage(frame);
        logger.debug("FrameDifferenceMotionDetector::prev_frame", prev_frame_);
    } else {
        current_frame_ = preProcessImage(frame);
        logger.debug("FrameDifferenceMotionDetector::current_frame", current_frame_);
    }
}


bool
FrameDifferenceMotionDetector::detectMotion(cv::Mat frame) {
    cv::Mat next_frame = preProcessImage(frame);

    cv::Mat diff1;
    cv::absdiff(prev_frame_, next_frame, diff1);
    logger.trace("FrameDifferenceMotionDetector::diff1", diff1);

    cv::Mat diff2;
    cv::absdiff(next_frame, current_frame_, diff2);
    logger.trace("FrameDifferenceMotionDetector::diff2", diff2);

    prev_frame_ = current_frame_;
    current_frame_ = next_frame;

    cv::Mat motion;
    cv::bitwise_and(diff1, diff2, motion);
    logger.debug("FrameDifferenceMotionDetector::bitwise_and", motion);

    // binarise
    cv::Mat thres;
    cv::threshold(motion, thres, MOTION_THRESH_MIN, MOTION_THRESH_MAX, cv::THRESH_BINARY);
    logger.trace("FrameDifferenceMotionDetector::threshold", thres);

    cv::Mat eroded;
    erode(thres, eroded, MOTION_ERODE_STRUCTURING);
    logger.debug("FrameDifferenceMotionDetector::eroded", eroded);

    /*
     * Determine number of changed pixels. Binarized image should only have values of 0 and 255
     * so we devide sum by 255 to get count of changed pixels.
     */
    cv::Scalar sum = cv::sum(eroded);
    double changed_pixels = sum.val[0] / 255;
    //std::cout << "changed_pixels = " << changed_pixels << std::endl;
    return changed_pixels > threshold_;
}


cv::Mat
FrameDifferenceMotionDetector::preProcessImage(cv::Mat frame) {
    cv::Mat input = frame;
    if (image_width_ > 0) {
        input = resizeToWidth(frame, image_width_);
    }
    cv::Mat grey;
    cv::cvtColor(input, grey, cv::COLOR_BGR2GRAY);
    if (use_blur_) {
        cv::Mat blurred;
        cv::GaussianBlur(grey, blurred, cv::Size(MOTION_BLUR_KERNEL_SIZE, MOTION_BLUR_KERNEL_SIZE), 0);
        return blurred;
    } else {
        return grey;
    }
}
