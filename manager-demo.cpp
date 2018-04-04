/*
 *  Face tracker 0.1
 *
 *  Copyright (c) 2018 David Snowdon. All rights reserved.
 *
 *  Distributed under the Boost Software License, Version 1.0. (See accompanying
 *  file LICENSE.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include "imagelogger.h"
#include "facedetector.h"
#include "manager.h"
#include "demo-util.h"

void usage() {
    std::cout << "Usage: <filename> [method] [[name face-image-filename]+]" << std::endl;
    std::cout << "Valid methods: NONE, CONTOURS, MSE, MSE_WITH_BLUR, DIFF, DIFF_WITH_BLUR" << std::endl;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        usage();
        return EXIT_FAILURE;
    }
    logger.enable(false);

    char *videoFilename = argv[1];
    std::string methodName = argv[2];
    MotionMethod method = motionMethodFromString(methodName);
    std::cout << "Read " << videoFilename << ", motion detector " << motionMethodToString(method) << std::endl;

    FaceDetector faceDetector("models");
    Manager *manager = new Manager(faceDetector);

    for (int f = 3; f < argc; f += 2) {
        std::string name = argv[f];
        std::string face_filename = argv[f + 1];
        std::cout << "Name: " << name << ", face: " << face_filename << std::endl;
        manager->addPerson(name, face_filename);
        // TODO load names & faces to train face detector
    }

    // Read video
    cv::VideoCapture video(videoFilename);

    // Exit if video is not opened
    if (!video.isOpened()) {
        std::cout << "Could not read video file" << std::endl;
        return EXIT_FAILURE;
    }

    // Camera sensor takes a while to calibrate, skip the first few frames
    for (int w = 0; w < WARM_UP_FRAMES; ++w) {
        cv::Mat drop_frame;
        video.read(drop_frame);
    }

    cv::Mat frame;
    cv::Mat prevFrame;
    int frameCount = 0;

    // Initialise detector
    MotionDetector *detector = motionDetectorFactory(method);
    int num_init_frames = detector->numInitFrames();
    for (int i = 0; i < num_init_frames; ++i) {
        video.read(prevFrame);
        detector->initFrame(prevFrame);
    }

    while (video.read(frame)) {
        ++frameCount;
        logger.nextFrame();
        bool moved = true;
        moved = detector->detectMotion(frame);

        if (moved) {
            manager->newFrame(frameCount, frame);
        }

        // TODO draw bounding boxes around faces
        // TODO draw names next to bounding boxes
        // TODO display frame rate
    }

    return EXIT_SUCCESS;
}