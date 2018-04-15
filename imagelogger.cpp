/*
 *  Face manager 0.1
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
ImageLogger::log(int msgLevel, std::string step, const cv::Mat &image) {
    if (enabled && (msgLevel >= logLevel)) {
        if (!logFile.is_open()) {
            firstLog();
        }
        cv::imwrite(filename(step), image);
        ++seq;
    }
}

void
ImageLogger::log(int msgLevel, std::string step, const dlib::matrix<dlib::rgb_pixel> &dlib_image) {
    if (enabled && (msgLevel >= logLevel)) {
        if (!logFile.is_open()) {
            firstLog();
        }
        dlib::save_png(dlib_image, filename(step));
        ++seq;
    }
}

void
ImageLogger::log(int msgLevel, std::string msg) {
    if (enabled && (msgLevel >= logLevel)) {
        if (!logFile.is_open()) {
            firstLog();
        }
        logFile << frameString() << levelString(msgLevel) << msg << std::endl;
    }
}

void
ImageLogger::firstLog() {
    if (enabled) {
        mkpath(dirName.c_str(), 0777);
        logFile.open(imagePrefix + DEFAULT_LOG_NAME);
        if (!logFile.is_open()) {
            std::cerr << "Failed to open " << (imagePrefix + DEFAULT_LOG_NAME) << std::endl;
        }
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

std::string
ImageLogger::levelString(int msgLevel) const {
    switch (msgLevel) {
        case L_TRACE:
            return " TRACE ";
        case L_DEBUG:
            return " DEBUG ";
        case L_INFO:
            return " INFO ";
        case L_ERROR:
            return " ERROR ";
        default:
            return " UNKNOWN ";
    }
}

