/*
 *  Face tracker 0.1
 *  Implementation of motion detector algorithms
 *
 *  Copyright (c) 2018 David Snowdon. All rights reserved.
 *
 *  Distributed under the Boost Software License, Version 1.0. (See accompanying
 *  file LICENSE.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */


#ifndef MOTION_DETECTOR_H_
#define MOTION_DETECTOR_H_

#include "imagelogger.h"

#include <opencv2/opencv.hpp>

/**
 * Motion detectors are stateful classes that detect motion in a sequence of images
 */
class MotionDetector {
public:
    /*
     * How many frames does the detector need before it can start detecting motion
     */
    virtual int numInitFrames() = 0;

    /*
     * Pass an initialization frame. This must be called numInitFrames() times
     */
    virtual void initFrame(cv::Mat frame) = 0;

    /*
     * After initialisation determine if this frame contains motion
     */
    virtual bool detectMotion(cv::Mat frame) = 0;
};


/** 
 * Implement a no-op detector to make it possible to create timings without motion detection
 */
class ConstantMotionDetector : public MotionDetector {
public:
    ConstantMotionDetector(bool detect_motion_result) {
        detect_motion_result_ = detect_motion_result;
    }

    virtual int numInitFrames() {
        return 0;
    }

    virtual void initFrame(cv::Mat frame) {
    }

    virtual bool detectMotion(cv::Mat frame) {
        return detect_motion_result_;
    }

private:
    bool detect_motion_result_;
};


/**
 * Based on algoritm described in https://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/
 */
class ContourMotionDetector : public MotionDetector {
public:
    ContourMotionDetector(int image_width, int motion_detected_area) {
        image_width_ = image_width;
        motion_dectected_area_ = motion_detected_area;
    }

    virtual int numInitFrames() {
        return 1;
    }

    virtual void initFrame(cv::Mat frame);

    virtual bool detectMotion(cv::Mat frame);

private:
    cv::Mat preProcessImage(cv::Mat frame);

    int image_width_;
    int motion_dectected_area_;
    cv::Mat accumulator_;
};


/**
 * Based on algorithm described in https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
 */
class MeanSquaredErrorMotionDetector : public MotionDetector {
public:
    MeanSquaredErrorMotionDetector(int image_width, double threshold, bool use_blur) {
        image_width_ = image_width;
        threshold_ = threshold;
        use_blur_ = use_blur;
    }

    virtual int numInitFrames() {
        return 1;
    }

    virtual void initFrame(cv::Mat frame);

    virtual bool detectMotion(cv::Mat frame);

private:
    cv::Mat preProcessImage(cv::Mat frame);

    int image_width_;
    double threshold_;
    bool use_blur_;
    cv::Mat accumulator_;
};


/**
 * Based on algorithm and code from
 * http://www.steinm.com/blog/motion-detection-webcam-python-opencv-differential-images/
 * https://github.com/cedricve/motion-detection
 */
class FrameDifferenceMotionDetector : public MotionDetector {
public:
    FrameDifferenceMotionDetector(int image_width, double threshold, bool use_blur) {
        image_width_ = image_width;
        threshold_ = threshold;
        use_blur_ = use_blur;
    }

    virtual int numInitFrames() {
        return 2;
    }

    virtual void initFrame(cv::Mat frame);

    virtual bool detectMotion(cv::Mat frame);

private:
    cv::Mat preProcessImage(cv::Mat frame);

    int image_width_;
    double threshold_;
    bool use_blur_;
    cv::Mat prev_frame_;
    cv::Mat current_frame_;
};


#endif // MOTION_DETECTOR_H_
