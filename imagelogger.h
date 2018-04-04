/*
 *  Face tracker 0.1
 *  Class used to help debug image processing applications
 *
 *  Copyright (c) 2018 David Snowdon. All rights reserved.
 *
 *  Distributed under the Boost Software License, Version 1.0. (See accompanying
 *  file LICENSE.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */


#ifndef IMAGELOGGER_H_
#define IMAGELOGGER_H_

#include <string>
#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>

int const L_TRACE = 0;
int const L_DEBUG = 1;
int const L_INFO = 2;
int const L_ERROR = 3;

class ImageLogger {
public:
    ImageLogger(std::string dir);

    ~ImageLogger();

    void enable(bool e) {
        enabled = e;
    }

    void level(int newLevel) {
        logLevel = newLevel;
    }

    void nextFrame() {
        ++frameCount;
        seq = 0;
    }

    void setFrame(int f) {
        frameCount = f;
        seq = 0;
    }

    inline bool traceEnabled() const {
        return enabled && L_TRACE >= logLevel;
    }

    inline bool debugEnabled() const {
        return enabled && L_DEBUG >= logLevel;
    }

    inline bool infoEnabled() const {
        return enabled && L_INFO >= logLevel;
    }

    inline bool errorEnabled() const {
        return enabled && L_ERROR >= logLevel;
    }

    template <typename image_type>
    inline void trace(std::string step, const image_type& image) {
        log(L_TRACE, step, image);
    }

    template <typename image_type>
    inline void debug(std::string step, const image_type& image) {
        log(L_DEBUG, step, image);
    }

    template <typename image_type>
    inline void info(std::string step, const image_type& image) {
        log(L_INFO, step, image);
    }

    template <typename image_type>
    inline void error(std::string step, const image_type& image) {
        log(L_ERROR, step, image);
    }

    void log(int msgLevel, std::string step, const cv::Mat& image);

    void log(int msgLevel, std::string step, const dlib::matrix<dlib::rgb_pixel>& dlib_image);

    inline void trace(std::string msg) {
        log(L_TRACE, msg);
    }

    inline void debug(std::string msg) {
        log(L_DEBUG, msg);
    }

    inline void info(std::string msg) {
        log(L_INFO, msg);
    }

    inline void error(std::string msg) {
        log(L_ERROR, msg);
    }

    void log(int msgLevel, std::string msg);

    // Convenience functions for common structures to log
    inline void trace(std::string msg, dlib::rectangle& rect) {
        log(L_TRACE, msg, rect);
    }

    inline void debug(std::string msg, dlib::rectangle& rect) {
        log(L_DEBUG, msg, rect);
    }

    inline void info(std::string msg, dlib::rectangle& rect) {
        log(L_INFO, msg, rect);
    }

    inline void error(std::string msg, dlib::rectangle& rect) {
        log(L_ERROR, msg, rect);
    }

    inline std::string to_string(dlib::rectangle& rect) {
        return std::to_string(rect.left()) + ", " +
               std::to_string(rect.top()) + ", " +
               std::to_string(rect.right()) + ", " +
               std::to_string(rect.bottom());
    }

    inline void log(int msgLevel, std::string msg, dlib::rectangle& rect) {
        log(L_ERROR, msg + to_string(rect));
    }

private:
    void firstLog();
    std::string filename(std::string step);
    std::string frameString();

    bool enabled = true;
    int logLevel = L_DEBUG;
    int frameCount = 0;
    int seq = 0;
    std::string dirName;
    std::string imagePrefix;
    std::ofstream logFile;
};

// define the single instance all modules will use
extern ImageLogger logger;

#endif // IMAGELOGGER_H_
