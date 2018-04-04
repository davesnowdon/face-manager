/*
 *  Face tracker 0.1
 *
 *  Copyright (c) 2018 David Snowdon. All rights reserved.
 *
 *  Distributed under the Boost Software License, Version 1.0. (See accompanying
 *  file LICENSE.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */


#ifndef FINAL_PROJECT_FACE_DETECTOR_H
#define FINAL_PROJECT_FACE_DETECTOR_H


#include <dlib/opencv.h>

// A face descriptor allows us to compare faces and determine if they are the same person
typedef dlib::matrix<float, 0, 1> FaceDescriptor;

class FaceDetectorImpl;

struct FaceCounters {
public:
    int detect_count_ = 0;
    int extract_face_image_count_ = 0;
    int face_descriptor_count_ = 0;

    inline void reset() {
        detect_count_ = 0;
        extract_face_image_count_ = 0;
        face_descriptor_count_ = 0;
    }
};

/*
 * Abstract the details of how we detect and identify faces from client code.
 * For now this contains the face detection, alignment and code to obtain deep metrics although later
 * may consider separating these functions.
 *
 * We don't make any attempt to hide the underlying dlib types either since the goal is not to provide
 * abstraction.
 */
class FaceDetector {
public:
    FaceDetector(const std::string &model_dir);
    ~FaceDetector();

    // TODO generalise the input image type
    std::vector<dlib::rectangle> detectFaces(const dlib::cv_image<dlib::bgr_pixel> &image);
    std::vector<dlib::rectangle> detectFaces(const dlib::array2d<dlib::rgb_pixel> &image);

    // TODO generalise the returned image type
    std::vector<dlib::matrix<dlib::rgb_pixel>> extractFaceImages(const dlib::cv_image<dlib::bgr_pixel> &image,
                                                                 const std::vector<dlib::rectangle> &face_bounds);

    // TODO generalise the returned image type
    dlib::matrix<dlib::rgb_pixel> extractFaceImage(const dlib::cv_image<dlib::bgr_pixel> &image,
                                                   const dlib::rectangle &face_bounds);
    dlib::matrix<dlib::rgb_pixel> extractFaceImage(const dlib::array2d<dlib::rgb_pixel> &image,
                                                   const dlib::rectangle &face_bounds);

    std::vector<FaceDescriptor> getFaceDescriptors(std::vector<dlib::matrix<dlib::rgb_pixel>> face_images);

    inline FaceDescriptor getFaceDescriptor(dlib::matrix<dlib::rgb_pixel> face_image) {
        return getFaceDescriptor(face_image, false);
    }

    /*
     * Jitter makes the calculated descriptor slightly more accurate by taking the mean of
     * sevral variants of the input image but is more computationally expensive.
     */
    FaceDescriptor getFaceDescriptor(dlib::matrix<dlib::rgb_pixel> face_image, bool use_jitter);

    void resetCounters() {
        counters_.reset();
    }

    FaceCounters getCounters() const {
        return counters_;
    }

private:
    // Avoid having to reference the back-end implementation in the header file. Otherwise we end up
    // putting the definition of the neural net in the header file which adds a lot of noise
    FaceDetectorImpl* impl;

    // counters so we can easily check later how much work we are doing
    FaceCounters counters_;
};


#endif //FINAL_PROJECT_FACE_DETECTOR_H
