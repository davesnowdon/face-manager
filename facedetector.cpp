/*
 *  Face manager 0.1
 *
 *  Copyright (c) 2018 David Snowdon. All rights reserved.
 *
 *  Distributed under the Boost Software License, Version 1.0. (See accompanying
 *  file LICENSE.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */


#include "facedetector.h"
#include "imagelogger.h"

#include <dlib/matrix.h>
#include <dlib/dnn.h>
#include <dlib/image_processing/frontal_face_detector.h>

// From dnn_face_recognition_ex.cpp
// ----------------------------------------------------------------------------------------

// The next bit of code defines a ResNet network.  It's basically copied
// and pasted from the dnn_imagenet_ex.cpp example, except we replaced the loss
// layer with loss_metric and made the network somewhat smaller.  Go read the introductory
// dlib DNN examples to learn what all this stuff means.
//
// Also, the dnn_metric_learning_on_images_ex.cpp example shows how to train this network.
// The dlib_face_recognition_resnet_model_v1 model used by this example was trained using
// essentially the code shown in dnn_metric_learning_on_images_ex.cpp except the
// mini-batches were made larger (35x15 instead of 5x5), the iterations without progress
// was set to 10000, and the training dataset consisted of about 3 million images instead of
// 55.  Also, the input layer was locked to images of size 150.
template<template<int, template<typename> class, int, typename> class block, int N,
        template<typename> class BN, typename SUBNET>
using residual = dlib::add_prev1<block<N, BN, 1, dlib::tag1<SUBNET>>>;

template<template<int, template<typename> class, int, typename> class block, int N,
        template<typename> class BN, typename SUBNET>
using residual_down = dlib::add_prev2<dlib::avg_pool<2, 2, 2, 2, dlib::skip1<dlib::tag2<block<N, BN, 2, dlib::tag1<SUBNET>>>>>>;

template<int N, template<typename> class BN, int stride, typename SUBNET>
using block  = BN<dlib::con<N, 3, 3, 1, 1, dlib::relu<BN<dlib::con<N, 3, 3, stride, stride, SUBNET>>>>>;

template<int N, typename SUBNET> using ares      = dlib::relu<residual<block, N, dlib::affine, SUBNET>>;
template<int N, typename SUBNET> using ares_down = dlib::relu<residual_down<block, N, dlib::affine, SUBNET>>;

template<typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
template<typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template<typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template<typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template<typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = dlib::loss_metric<dlib::fc_no_bias<128, dlib::avg_pool_everything<
        alevel0<alevel1<alevel2<alevel3<alevel4<dlib::max_pool<3, 3, 2, 2, dlib::relu<dlib::affine<dlib::con<32, 7, 7, 2, 2,
                dlib::input_rgb_image_sized<150>
        >>>>>>>>>>>>;

// ----------------------------------------------------------------------------------------

class FaceDetectorImpl {
public:
    FaceDetectorImpl(const std::string &model_dir);

    // TODO generalise the input image type
    std::vector<dlib::rectangle> detectFaces(const dlib::cv_image<dlib::bgr_pixel> &image);

    std::vector<dlib::rectangle> detectFaces(const dlib::array2d<dlib::rgb_pixel> &image);

    // TODO generalise the returned image type
    std::vector<dlib::matrix<dlib::rgb_pixel>> extractFaceImages(const dlib::cv_image<dlib::bgr_pixel> &image,
                                                                 const std::vector<dlib::rectangle> &face_bounds) const;

    dlib::matrix<dlib::rgb_pixel> extractFaceImage(const dlib::cv_image<dlib::bgr_pixel> &image,
                                                   const dlib::rectangle &face_bounds) const;

    dlib::matrix<dlib::rgb_pixel> extractFaceImage(const dlib::array2d<dlib::rgb_pixel> &image,
                                                   const dlib::rectangle &face_bounds) const;

    std::vector<FaceDescriptor> getFaceDescriptors(std::vector<dlib::matrix<dlib::rgb_pixel>> face_images);

    FaceDescriptor getFaceDescriptor(const dlib::matrix<dlib::rgb_pixel> &face_image, bool use_jitter);

private:
    // Get the face detector
    dlib::frontal_face_detector face_detector = dlib::get_frontal_face_detector();

    // facial landmark detector
    dlib::shape_predictor landmark_detector;

    // DNN used for face recognition
    anet_type face_metrics_net;
};


FaceDetectorImpl::FaceDetectorImpl(const std::string &model_dir) {
    // Get the face detector
    face_detector = dlib::get_frontal_face_detector();

    // facial landmark detector
    dlib::deserialize(model_dir + "/shape_predictor_5_face_landmarks.dat") >> landmark_detector;

    // DNN used for face recognition
    dlib::deserialize(model_dir + "/dlib_face_recognition_resnet_model_v1.dat") >> face_metrics_net;
}


std::vector<dlib::rectangle>
FaceDetectorImpl::detectFaces(const dlib::cv_image<dlib::bgr_pixel> &image) {
    return face_detector(image);
}

std::vector<dlib::rectangle>
FaceDetectorImpl::detectFaces(const dlib::array2d<dlib::rgb_pixel> &image) {
    return face_detector(image);
}


std::vector<dlib::matrix<dlib::rgb_pixel>>
FaceDetectorImpl::extractFaceImages(const dlib::cv_image<dlib::bgr_pixel> &image,
                                    const std::vector<dlib::rectangle> &face_bounds) const {
    // These are the transformed and extracted faces
    std::vector<dlib::matrix<dlib::rgb_pixel>> faces;

    // Loop over all detected face rectangles
    for (const auto &face_bound : face_bounds) {
        faces.push_back(std::move(extractFaceImage(image, face_bound)));
    }
    return faces;
}


// TODO fix duplication
dlib::matrix<dlib::rgb_pixel>
FaceDetectorImpl::extractFaceImage(const dlib::cv_image<dlib::bgr_pixel> &image,
                                   const dlib::rectangle &face_bounds) const {
    // Find the face landmarks
    dlib::full_object_detection landmarks = landmark_detector(image, face_bounds);

    // use the landmarks to normalise the face image and extract
    dlib::matrix<dlib::rgb_pixel> face_chip;
    dlib::extract_image_chip(image, dlib::get_face_chip_details(landmarks, 150, 0.25), face_chip);
    logger.debug("face-chip", face_chip);
    return face_chip;
}

dlib::matrix<dlib::rgb_pixel>
FaceDetectorImpl::extractFaceImage(const dlib::array2d<dlib::rgb_pixel> &image,
                                   const dlib::rectangle &face_bounds) const {
    // Find the face landmarks
    dlib::full_object_detection landmarks = landmark_detector(image, face_bounds);

    // use the landmarks to normalise the face image and extract
    dlib::matrix<dlib::rgb_pixel> face_chip;
    dlib::extract_image_chip(image, dlib::get_face_chip_details(landmarks, 150, 0.25), face_chip);
    logger.debug("face-chip", face_chip);
    return face_chip;
}

std::vector<FaceDescriptor>
FaceDetectorImpl::getFaceDescriptors(std::vector<dlib::matrix<dlib::rgb_pixel>> face_images) {
    return face_metrics_net(face_images);
}


// From dnn_face_recognition_ex.cpp
std::vector<dlib::matrix<dlib::rgb_pixel>>
jitter_image(const dlib::matrix<dlib::rgb_pixel> &img) {
    // All this function does is make 100 copies of img, all slightly jittered by being
    // zoomed, rotated, and translated a little bit differently. They are also randomly
    // mirrored left to right.
    thread_local dlib::rand rnd;

    std::vector<dlib::matrix<dlib::rgb_pixel>> crops;
    for (int i = 0; i < 100; ++i)
        crops.push_back(dlib::jitter_image(img, rnd));

    return crops;
}

FaceDescriptor
FaceDetectorImpl::getFaceDescriptor(const dlib::matrix<dlib::rgb_pixel> &face_image, bool use_jitter) {
    if (use_jitter) {
        return dlib::mean(dlib::mat(face_metrics_net(jitter_image(face_image))));
    } else {
        std::vector<dlib::matrix<dlib::rgb_pixel>> face_images{face_image};
        auto descriptors = face_metrics_net(face_images);
        return descriptors[0];
    }
}


FaceDetector::FaceDetector(const std::string &model_dir) {
    impl = new FaceDetectorImpl(model_dir);
}

FaceDetector::~FaceDetector() {
    delete impl;
}

std::vector<dlib::rectangle>
FaceDetector::detectFaces(const dlib::cv_image<dlib::bgr_pixel> &image) {
    ++counters_.detect_count_;
    return impl->detectFaces(image);
}

std::vector<dlib::rectangle>
FaceDetector::detectFaces(const dlib::array2d<dlib::rgb_pixel> &image) {
    ++counters_.detect_count_;
    return impl->detectFaces(image);
}


std::vector<dlib::matrix<dlib::rgb_pixel>>
FaceDetector::extractFaceImages(const dlib::cv_image<dlib::bgr_pixel> &image,
                                const std::vector<dlib::rectangle> &face_bounds) {
    counters_.extract_face_image_count_ += face_bounds.size();
    return impl->extractFaceImages(image, face_bounds);
}


dlib::matrix<dlib::rgb_pixel>
FaceDetector::extractFaceImage(const dlib::cv_image<dlib::bgr_pixel> &image,
                               const dlib::rectangle &face_bounds) {
    ++counters_.extract_face_image_count_;
    return impl->extractFaceImage(image, face_bounds);
}

dlib::matrix<dlib::rgb_pixel>
FaceDetector::extractFaceImage(const dlib::array2d<dlib::rgb_pixel> &image,
                               const dlib::rectangle &face_bounds) {
    ++counters_.extract_face_image_count_;
    return impl->extractFaceImage(image, face_bounds);
}


std::vector<FaceDescriptor>
FaceDetector::getFaceDescriptors(std::vector<dlib::matrix<dlib::rgb_pixel>> face_images) {
    counters_.face_descriptor_count_ += face_images.size();
    return impl->getFaceDescriptors(face_images);
}


FaceDescriptor
FaceDetector::getFaceDescriptor(const dlib::matrix<dlib::rgb_pixel> face_image, bool use_jitter) {
    ++counters_.face_descriptor_count_;
    return impl->getFaceDescriptor(face_image, use_jitter);
}