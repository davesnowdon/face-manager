/*
 *  Face manager 0.1
 *
 *  Copyright (c) 2018 David Snowdon. All rights reserved.
 *
 *  Distributed under the Boost Software License, Version 1.0. (See accompanying
 *  file LICENSE.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include <cmath>
#include <iomanip>

#include "imagelogger.h"
#include "facedetector.h"
#include "manager.h"
#include "demo-util.h"


const double FPS_MOVING_AVERAGE_WEIGHT = 0.9;
const cv::Point FPS_TEXT_POSITION(20, 20);
const int FPS_TEXT_FONT = cv::FONT_HERSHEY_SIMPLEX;
const double FPS_TEXT_SCALE = 0.75;
const cv::Scalar FPS_TEXT_COLOUR(255, 128, 0);
const int FPS_TEXT_THICKNESS = 2;
const char *FPS_TEXT_PREFIX = "FPS: ";
const int FPS_WIDTH = 5;
const int FPS_PRECISION = 3;

const char *VISIBLE_COUNT_PREFIX = ", #visible: ";
const int VISIBLE_COUNT_WIDTH = 2;
const char *KNOWN_COUNT_PREFIX = ", #people: ";
const int KNOWN_COUNT_WIDTH = 2;


const char *PERSON_UNKNOWN_PREFIX = "Local ID: ";
const cv::Scalar PERSON_NAME_COLOUR(255, 0, 0);
const int PERSON_NAME_FONT = cv::FONT_HERSHEY_SIMPLEX;
const double PERSON_NAME_SCALE = 0.75;
const int PERSON_NAME_THICKNESS = 2;
const cv::Scalar PERSON_BOX_COLOUR(255, 0, 0);
const int PERSON_BOX_THICKNESS = 2;

void usage() {
    std::cout
            << "Takes an input video file and annotates it with fae tracking results and frame rate and writes output to another video file"
            << std::endl;
    std::cout << "Usage: <input filename> <output filename> [method] [[name face-image-filename]+]" << std::endl;
    std::cout << "Valid methods: NONE, CONTOURS, MSE, MSE_WITH_BLUR, DIFF, DIFF_WITH_BLUR" << std::endl;
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

int main(int argc, char **argv) {
    if (argc < 4) {
        usage();
        return EXIT_FAILURE;
    }
    logger.setFrame(0);
    logger.enable(true);

    char *inputVideoFilename = argv[1];
    char *outputVideoFilename = argv[2];
    std::string methodName = argv[3];
    MotionMethod method = motionMethodFromString(methodName);
    std::cout << "Read " << inputVideoFilename << ", write " << outputVideoFilename << ", motion detector "
              << motionMethodToString(method) << std::endl;

    FaceDetector faceDetector("models");
    Manager *manager = new Manager(faceDetector);

    for (int f = 4; f < argc; f += 2) {
        std::string name = argv[f];
        std::string face_filename = argv[f + 1];
        std::cout << "Name: " << name << ", face: " << face_filename << std::endl;
        manager->addPerson(name, face_filename);
    }

    // Read video
    cv::VideoCapture input_video(inputVideoFilename);

    // Exit if video is not opened
    if (!input_video.isOpened()) {
        std::cout << "Could not read video file" << std::endl;
        return EXIT_FAILURE;
    }

    // save the input width & height from the input so we can use the same for the output file
    int frame_width = input_video.get(CV_CAP_PROP_FRAME_WIDTH);
    int frame_height = input_video.get(CV_CAP_PROP_FRAME_HEIGHT);
    int input_fps = input_video.get(CV_CAP_PROP_FPS);
    int input_codec = static_cast<int>(input_video.get(CV_CAP_PROP_FOURCC));

    // Create the output file
    std::cout << "Writing to " << outputVideoFilename
              << " with size " << frame_width << " x " << frame_height
              << " at " << input_fps << " FPS" << std::endl;
    cv::VideoWriter output_video(outputVideoFilename, CV_FOURCC('M', 'J', 'P', 'G'), input_fps,
                                 cv::Size(frame_width, frame_height));

    // Camera sensor takes a while to calibrate, skip the first few frames
    for (int w = 0; w < WARM_UP_FRAMES; ++w) {
        cv::Mat drop_frame;
        input_video.read(drop_frame);
    }

    cv::Mat frame;
    cv::Mat prevFrame;
    int frameCount = 0;

    // Initialise detector
    MotionDetector *detector = motionDetectorFactory(method);
    int num_init_frames = detector->numInitFrames();
    for (int i = 0; i < num_init_frames; ++i) {
        input_video.read(prevFrame);
        detector->initFrame(prevFrame);
    }

    double meanFrameTime = 0;
    double minFps = std::numeric_limits<double>::max();
    double maxFps = 0;
    double startTicks = (double) cv::getTickCount();
    double lastFrame = startTicks;

    while (input_video.read(frame)) {
        ++frameCount;
        logger.nextFrame();
        bool moved = detector->detectMotion(frame);

        if (moved) {
            manager->newFrame(frameCount, frame);
        }

        auto visible_people = manager->visiblePeople();
        for (auto person : visible_people) {
            // draw box around tracked person
            dlib::rectangle bb = person->boundingBox();
            cv::rectangle(frame, dlibRectangleToOpenCV(bb), PERSON_BOX_COLOUR, PERSON_BOX_THICKNESS);

            // draw person's identifier next to bounding box
            std::string name = person->externalId();
            if (0 == name.length()) {
                // If no external ID known, just use local ID
                std::stringstream nameStream;
                nameStream << PERSON_UNKNOWN_PREFIX << person->localId();
                name = nameStream.str();
            }
            cv::Point namePos(bb.left(), bb.top());
            cv::putText(frame, name, namePos, PERSON_NAME_FONT, PERSON_NAME_SCALE, PERSON_NAME_COLOUR,
                        PERSON_NAME_THICKNESS);
        }

        // Compute exponentially weighted moving average (with bias correction) of frame rate
        double now = (double) cv::getTickCount();
        double thisFrame = now - lastFrame;
        lastFrame = now;
        double biasCorrectedMeanFrameTime = meanFrameTime / (1.0 - std::pow(FPS_MOVING_AVERAGE_WEIGHT, frameCount));
        meanFrameTime = (FPS_MOVING_AVERAGE_WEIGHT * biasCorrectedMeanFrameTime) +
                        (1.0 - FPS_MOVING_AVERAGE_WEIGHT) * thisFrame;
        double fps = cv::getTickFrequency() / meanFrameTime;
        minFps = std::min(minFps, fps);
        maxFps = std::max(maxFps, fps);

        // Display frames per second and other info
        std::stringstream fpsText;
        fpsText << std::setprecision(FPS_PRECISION)
                << FPS_TEXT_PREFIX << std::setw(FPS_WIDTH) << fps
                << VISIBLE_COUNT_PREFIX << std::setw(VISIBLE_COUNT_WIDTH) << manager->visibleCount()
                << KNOWN_COUNT_PREFIX << std::setw(KNOWN_COUNT_WIDTH) << manager->knownCount();
        cv::putText(frame, fpsText.str(), FPS_TEXT_POSITION, FPS_TEXT_FONT, FPS_TEXT_SCALE, FPS_TEXT_COLOUR,
                    FPS_TEXT_THICKNESS);

        output_video.write(frame);
    }

    double endTicks = (double) cv::getTickCount();
    double durationTicks = endTicks - startTicks;
    double meanTicks = durationTicks / frameCount;
    double meanFps = cv::getTickFrequency() / meanTicks;

    input_video.release();
    output_video.release();

    std::cout << "Mean FPS " << meanFps << ", Min FPS " << minFps << ", Max FPS " << maxFps << std::endl;

    return EXIT_SUCCESS;
}


