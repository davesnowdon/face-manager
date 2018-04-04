/*
 *  Face tracker 0.1
 *
 *  Copyright (c) 2018 David Snowdon. All rights reserved.
 *
 *  Distributed under the Boost Software License, Version 1.0. (See accompanying
 *  file LICENSE.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */


#include "imagelogger.h"
#include "mkpath.h"
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <iostream>

const std::string DEFAULT_LOG_NAME = "log.txt";

// The single instance of the logger
ImageLogger logger("debug");

ImageLogger::ImageLogger(std::string dir) {
    dirName = dir;
    imagePrefix = dir + "/";
}

ImageLogger::~ImageLogger() {
   if (logFile) {
       logFile.close();
   }
}

void
ImageLogger::log(int msgLevel, std::string step, const cv::Mat& image) {
    if (enabled && (msgLevel >= logLevel)) {
        if (0 == seq) {
            firstLog();
        }
        cv::imwrite(filename(step), image);
        ++seq;
    }
}

void
ImageLogger::log(int msgLevel, std::string step, const dlib::matrix<dlib::rgb_pixel>& dlib_image) {
    if (enabled && (msgLevel >= logLevel)) {
        if (0 == seq) {
            firstLog();
        }
        dlib::save_png(dlib_image, filename(step));
        ++seq;
    }
}

void
ImageLogger::log(int msgLevel, std::string msg) {
    if (enabled && (msgLevel >= logLevel)) {
        if (0 == seq) {
            firstLog();
        }
        logFile << msg << std::endl;
    }
}

void
ImageLogger::firstLog() {
    if (enabled) {
        mkpath(dirName.c_str(), 0777);
        logFile.open(imagePrefix + DEFAULT_LOG_NAME);
    }
}

std::string
ImageLogger::filename(std::string step) {
    return imagePrefix + frameString() + "-" + step + ".png";
}

std::string
ImageLogger::frameString() {
    char numbers[25];
    sprintf(numbers, "%05d-%03d", frameCount, seq);
    return std::string(numbers);
}



