/*
 *  Face manager 0.1
 *
 *  Copyright (c) 2018 David Snowdon. All rights reserved.
 *
 *  Distributed under the Boost Software License, Version 1.0. (See accompanying
 *  file LICENSE.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include "imagelogger.h"
#include "motiondetector.h"
#include "facedetector.h"
#include "manager.h"
#include "demo-util.h"

#include <stdlib.h>
#include <cstring>
#include <opencv2/opencv.hpp>

#include <dlib/dnn.h>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/image_transforms.h>

enum ProcessingType {
    NAIVE,   // Use a naive approach that runs face detection every N frames
    MANAGER, // Use the face manager which uses a mix of detection and tracking
    NONE     // Do no further processing so we can isolate the cost of doing only motion detection
};

char const*
processingTypeToString(ProcessingType processingType) {
    switch (processingType) {
        case ProcessingType::NONE:
            return "NONE";

        case ProcessingType::NAIVE:
            return "NAIVE";

        case ProcessingType::MANAGER:
            return "MANAGER";
        default:
            return "";
    }
}

void usage() {
    std::cout << "Usage: <filename> <iterations> [method]" << std::endl;
    std::cout << "Valid methods: NONE, CONTOURS, MSE, MSE_WITH_BLUR, DIFF, DIFF_WITH_BLUR" << std::endl;
}

int
runTrial(MotionMethod method, int numIterations, char *videoFilename, bool enable_logging, bool enable_output,
         ProcessingType processingType, FaceDetector &faceDetector, Manager *manager) {
    if (enable_output) {
        std::cout << "Start: " << motionMethodToString(method) << ", logging enabled " << enable_logging << std::endl;
    }

    /*
     * For better accuracy we average time over the total number of frames
     * for the specified number of iterations
     */
    double totalTime = 0;
    int frameCount = 0;
    int motionCount = 0;
    for (int i = 0; i < numIterations; ++i) {

        // Read video
        cv::VideoCapture video(videoFilename);

        // Exit if video is not opened
        if (!video.isOpened()) {
            std::cout << "Could not read video file" << std::endl;
            return EXIT_FAILURE;
        }

        cv::Mat frame;
        cv::Mat prevFrame;

        // setup run
        frameCount = 0;
        motionCount = 0;
        logger.setFrame(0);
        logger.enable(0 == i && enable_logging);

        if (manager) {
            manager->reset();
        }

        // Camera sensor takes a while to calibrate, skip the first few frames
        for (int w = 0; w < WARM_UP_FRAMES; ++w) {
            cv::Mat drop_frame;
            video.read(drop_frame);
        }

        // Initialise detector
        MotionDetector *detector = motionDetectorFactory(method);
        int num_init_frames = detector->numInitFrames();
        for (int i = 0; i < num_init_frames; ++i) {
            video.read(prevFrame);
            detector->initFrame(prevFrame);
        }

        // we count the operations performed by the face detector as a measure of how much work we are doing
        faceDetector.resetCounters();

        // don't want to include setup time so start timing now
        double startTime = (double) cv::getTickCount();
        while (video.read(frame)) {
            ++frameCount;
            logger.nextFrame();
            bool moved = true;
            moved = detector->detectMotion(frame);

            if (moved) {
                ++motionCount;
                logger.info("motion", frame);
            }

            if (moved) {
                switch (processingType) {
                    case ProcessingType::NONE:
                        break;

                    case ProcessingType::NAIVE:
                        if (!manager) {
                            // Convert OpenCV image format to Dlib's image format
                            dlib::cv_image<dlib::bgr_pixel> frame_dlib(frame);

                            // Detect faces in the image
                            std::vector<dlib::rectangle> faceRects = faceDetector.detectFaces(frame_dlib);
                            if (logger.debugEnabled()) {
                                logger.debug("Number of faces detected: " + std::to_string(faceRects.size()));
                            }

                            // These are the transformed and extracted faces
                            std::vector<dlib::matrix<dlib::rgb_pixel>> faces = faceDetector.extractFaceImages(
                                    frame_dlib,
                                    faceRects);

                            if (faces.size() > 0) {
                                /*
                                 * This call asks the DNN to convert each face image in faces into a 128D vector.
                                 * In this 128D vector space, images from the same person will be close to each other
                                 * but vectors from different people will be far apart.  So we can use these vectors to
                                 * identify if a pair of images are from the same person or from different people.
                                 */
                                std::vector<dlib::matrix<float, 0, 1>> face_descriptors = faceDetector.getFaceDescriptors(
                                        faces);
                            }
                        }
                        break;

                    case ProcessingType::MANAGER:
                        if (manager) {
                            manager->newFrame(frameCount, frame);
                        }
                        break;
                }
            }

            prevFrame = frame;
        }
        totalTime += ((double) cv::getTickCount() - startTime);
    }


    // Calculate Frames per second (FPS)
    float fps = cv::getTickFrequency() / (totalTime / (frameCount * numIterations));
    FaceCounters counters = faceDetector.getCounters();
    if (enable_output) {
        std::cout
                << "File, method, Manager?, Detect inteval, #frames, FPS, #motion frames, #face detect, #face extract, #face descriptor"
                <<
                std::endl;
        std::cout << "End: " << videoFilename << ", "
                  << motionMethodToString(method)
                  << ", " << processingTypeToString(processingType)
                  << ", " << ((nullptr == manager) ? "" : std::to_string(manager->detectorFrameInterval()))
                  << ", " << frameCount << ", " << fps << ", " << motionCount
                  << ", " << counters.detect_count_
                  << ", " << counters.extract_face_image_count_
                  << ", " << counters.face_descriptor_count_
                  << std::endl;
    }

    return 0;
}


int
runMethods(int numIterations, char *videoFilename, ProcessingType processingType, FaceDetector &faceDetector,
           Manager *manager) {
    MotionMethod methods[] = {MOTION_ALWAYS, MOTION_NEVER,
                              MOTION_EVERY_OTHER, MOTION_EVERY_TEN,
                              MOTION_CONTOURS,
                              MOTION_MSE, MOTION_MSE_WITH_BLUR,
                              MOTION_DIFF, MOTION_DIFF_WITH_BLUR};
    for (const MotionMethod method : methods) {
        int result = runTrial(method, numIterations, videoFilename, false, true, processingType, faceDetector, manager);
        if (0 != result) {
            std::cerr << "Stopping early due to error" << std::endl;
            return result;
        }
    }
    return 0;
}


int main(int argc, char **argv) {
    if (argc < 3) {
        usage();
        return EXIT_FAILURE;
    }

    char *videoFilename = argv[1];
    int numIterations = atoi(argv[2]);
    std::cout << "Read " << videoFilename << " " << numIterations << " times" << std::endl;

    FaceDetector faceDetector("models");

    // Run a single iteration to "warm up" the system
    std::cout << "Start warm up" << std::endl;
    runTrial(MOTION_ALWAYS, 1, videoFilename, false, false, ProcessingType::NAIVE, faceDetector, nullptr);
    std::cout << "Warm up done" << std::endl;

    /*
     * Depending on whether the 3rd argument is given we will try all methods without logging or run
     * a single method with logging
     */
    if (4 == argc) {
        std::string methodName = argv[3];
        MotionMethod method = motionMethodFromString(methodName);
        // TODO add manager
        return runTrial(method, numIterations, videoFilename, true, true, ProcessingType::NAIVE, faceDetector, nullptr);

    } else {
        // run complete set of trials
        std::cout << "Running all methods using only motion detection" << std::endl;
        int result = runMethods(numIterations, videoFilename, ProcessingType::NONE, faceDetector, nullptr);

        if (0 == result) {
            std::cout << "Running all methods using naive approach" << std::endl;
            result = runMethods(numIterations, videoFilename, ProcessingType::NAIVE, faceDetector, nullptr);
        }

        if (0 == result) {
            std::cout << "Running all methods with manager (interval 5)" << std::endl;
            Manager *manager = new Manager(faceDetector);
            manager->detectorFrameInterval(5);
            result = runMethods(numIterations, videoFilename, ProcessingType::MANAGER, faceDetector, manager);
            delete manager;
        }

        if (0 == result) {
            std::cout << "Running all methods with manager (interval 10)" << std::endl;
            Manager *manager = new Manager(faceDetector);
            manager->detectorFrameInterval(10);
            result = runMethods(numIterations, videoFilename, ProcessingType::MANAGER, faceDetector, manager);
            delete manager;
        }

        return result;
    }

    return EXIT_SUCCESS;
}

