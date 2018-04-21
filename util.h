/*
 *  Face manager 0.1
 *
 *  Copyright (c) 2018 David Snowdon. All rights reserved.
 *
 *  Distributed under the Boost Software License, Version 1.0. (See accompanying
 *  file LICENSE.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FACE_MANAGER_UTIL_H
#define FACE_MANAGER_UTIL_H

#include <map>
#include <set>
#include <string>
#include <sstream>

#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>

// adapted from http://www.lonecpluspluscoder.com/2015/08/13/an-elegant-way-to-extract-keys-from-a-c-map/
template<typename TK, typename TV>
std::set<TK> extract_keys(std::map<TK, TV> const &input_map) {
    std::set<TK> retval;
    for (auto const &element: input_map) {
        retval.insert(element.first);
    }
    return retval;
}

template<typename TK, typename TV>
std::set<TV> extract_values(std::map<TK, TV> const &input_map) {
    std::set<TV> retval;
    for (auto const &element: input_map) {
        retval.insert(element.second);
    }
    return retval;
}

// TODO use type traits to ensure that C is a container with the value type TV
// https://stackoverflow.com/questions/7728478/c-template-class-function-with-arbitrary-container-type-how-to-define-it
template<typename TK, typename TV, typename C>
C extract_values(std::map<TK, TV> const &input_map, C &container) {
    auto inserter = std::inserter(container, container.end());
    for (auto const &element: input_map) {
        inserter = element.second;
    }
    return container;
}

template<typename T>
std::string set_to_string(std::set<T> const &input, std::string const &separator) {
    std::ostringstream oss;
    for (auto val : input) {
        oss << val << separator;
    }
    return oss.str();
}

/*
 * Rectangle conversion functions from https://stackoverflow.com/questions/34871740/convert-opencvs-rect-to-dlibs-rectangle
 */
static cv::Rect dlibRectangleToOpenCV(dlib::rectangle r) {
    return cv::Rect(cv::Point2i(r.left(), r.top()), cv::Point2i(r.right() + 1, r.bottom() + 1));
}

static dlib::rectangle openCVRectToDlib(cv::Rect r) {
    return dlib::rectangle((long) r.tl().x, (long) r.tl().y, (long) r.br().x - 1, (long) r.br().y - 1);
}

#endif //FACE_MANAGER_UTIL_H
