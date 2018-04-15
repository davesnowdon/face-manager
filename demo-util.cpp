/*
 *  Face manager 0.1
 *
 *  Copyright (c) 2018 David Snowdon. All rights reserved.
 *
 *  Distributed under the Boost Software License, Version 1.0. (See accompanying
 *  file LICENSE.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */


#include "demo-util.h"

#include <iostream>

MotionDetector *motionDetectorFactory(MotionMethod method) {
    switch (method) {
        case MOTION_ALWAYS:
            return new ConstantMotionDetector(true);
        case MOTION_NEVER:
            return new ConstantMotionDetector(false);
        case MOTION_CONTOURS:
            return new ContourMotionDetector(MOTION_WIDTH, MOTION_CONTOUR_MIN_AREA);
        case MOTION_MSE:
            return new MeanSquaredErrorMotionDetector(MOTION_WIDTH, MOTION_MSE_THRESHOLD, false);
        case MOTION_MSE_WITH_BLUR:
            return new MeanSquaredErrorMotionDetector(MOTION_WIDTH, MOTION_MSE_THRESHOLD, true);
        case MOTION_DIFF:
            return new FrameDifferenceMotionDetector(MOTION_WIDTH, MOTION_DIFF_THRESHOLD, false);
        case MOTION_DIFF_WITH_BLUR:
            return new FrameDifferenceMotionDetector(MOTION_WIDTH, MOTION_DIFF_THRESHOLD, true);
        default:
            std::cerr << "invalid motion detector type" << std::endl;
            std::exit(1);
    }
}


std::string stringToUpper(std::string to_convert) {
    std::transform(to_convert.begin(), to_convert.end(), to_convert.begin(), ::toupper);
    return to_convert;
}


MotionMethod motionMethodFromString(std::string method_name) {
    method_name = stringToUpper(method_name);
    if (method_name == "ALWAYS") {
        return MOTION_ALWAYS;
    } else if (method_name == "NEVER") {
        return MOTION_NEVER;
    } else if (method_name == "CONTOURS") {
        return MOTION_CONTOURS;
    } else if (method_name == "MSE") {
        return MOTION_MSE;
    } else if (method_name == "MSE_WITH_BLUR") {
        return MOTION_MSE_WITH_BLUR;
    } else if (method_name == "DIFF") {
        return MOTION_DIFF;
    } else if (method_name == "DIFF_WITH_BLUR") {
        return MOTION_DIFF_WITH_BLUR;
    } else {
        std::cerr << "invalid motion detector type: '" << method_name << "'" << std::endl;
        std::exit(1);
    }
}


std::string motionMethodToString(MotionMethod method) {
    switch (method) {
        case MOTION_ALWAYS:
            return "ALWAYS";
        case MOTION_NEVER:
            return "NEVER";
        case MOTION_CONTOURS:
            return "CONTOURS";
        case MOTION_MSE:
            return "MSE";
        case MOTION_MSE_WITH_BLUR:
            return "MSE_WITH_BLUR";
        case MOTION_DIFF:
            return "DIFF";
        case MOTION_DIFF_WITH_BLUR:
            return "DIFF_WITH_BLUR";
        default:
            return "";
    }
}
