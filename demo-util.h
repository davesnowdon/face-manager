/*
 *  Face tracker 0.1
 *
 *  Copyright (c) 2018 David Snowdon. All rights reserved.
 *
 *  Distributed under the Boost Software License, Version 1.0. (See accompanying
 *  file LICENSE.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */


#ifndef FINAL_PROJECT_DEMO_UTIL_H
#define FINAL_PROJECT_DEMO_UTIL_H

#include "motiondetector.h"

enum MotionMethod {
    MOTION_ALWAYS,   // always "detect" motion, used for comparison
    MOTION_NEVER,    // never detect motion, used to allow us to see cost of reading video with no processing overhead
    MOTION_CONTOURS, // Use the PyImageSearch method based on contours
    MOTION_MSE,      // Use mean squared error
    MOTION_MSE_WITH_BLUR, // Use mean squared error after blurring
    MOTION_DIFF,     // use frame differencing
    MOTION_DIFF_WITH_BLUR // use frame differencing after blurring
};

int const WARM_UP_FRAMES = 5;
int const MOTION_WIDTH = 500;
double const MOTION_CONTOUR_MIN_AREA = 500;
double const MOTION_MSE_THRESHOLD = 2000;
double const MOTION_DIFF_THRESHOLD = 250;

// ----------------------------------------------------------------------------------------

MotionDetector *motionDetectorFactory(MotionMethod method);

std::string stringToUpper(std::string to_convert);

MotionMethod motionMethodFromString(std::string method_name);

std::string motionMethodToString(MotionMethod method);

#endif //FINAL_PROJECT_DEMO_UTIL_H
