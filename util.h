/*
 *  Face tracker 0.1
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

template<typename T>
std::string set_to_string(std::set<T> const &input, std::string const &separator) {
    std::ostringstream oss;
    for (auto val : input) {
        oss << val << separator;
    }
    return oss.str();
}

#endif //FACE_MANAGER_UTIL_H
