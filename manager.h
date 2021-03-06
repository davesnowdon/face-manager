/*
 *  Face manager 0.1
 *
 *  Copyright (c) 2018 David Snowdon. All rights reserved.
 *
 *  Distributed under the Boost Software License, Version 1.0. (See accompanying
 *  file LICENSE.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef FINAL_PROJECT_MANAGER_H
#define FINAL_PROJECT_MANAGER_H

#include <string>
#include <memory>
#include <dlib/dnn.h>
#include <dlib/image_processing.h>

#include "facedetector.h"

//  Note that in dlib there is no explicit image object, just a 2D array and
// various pixel types. For readability we define an image type here.
typedef dlib::array2d<dlib::rgb_pixel> Image;

// Represents a tracked person (face). For now we only track faces.
class Person {
public:

    Person(int id, const dlib::rectangle &bounding_box, const Image &face_image, double blur,
           const FaceDescriptor &descriptor)
            : local_id_(id), bounding_box_(bounding_box), face_blur_(blur), face_descriptor_(descriptor) {
        dlib::assign_image(face_image_, face_image);
    }

    int localId() const {
        return local_id_;
    }

    const std::string externalId() const {
        return external_id_;
    }

    void externalId(const std::string new_id) {
        external_id_ = new_id;
    }

    const dlib::rectangle &boundingBox() const {
        return bounding_box_;
    }

    void boundingBox(const dlib::rectangle &new_box) {
        bounding_box_ = new_box;
    }

    const FaceDescriptor &faceDescriptor() const {
        return face_descriptor_;
    }

    void faceDescriptor(const FaceDescriptor &new_descriptor) {
        face_descriptor_ = new_descriptor;
    }

    const Image &faceImage() const {
        return face_image_;
    }

    void faceImage(const Image &new_image) {
        dlib::assign_image(face_image_, new_image);
    }

    double faceBlur() const {
        return face_blur_;
    }

    void faceBlur(double new_blur) {
        face_blur_ = new_blur;
    }

    int nonVisibleFrames() const {
        return non_visible_frames_;
    }

    void resetNonVisibleFrames() {
        non_visible_frames_ = 0;
    }

    int incNonVisibleFrames() {
        return ++non_visible_frames_;
    }

private:
    // Identifier local to this session that only applies within the current "session"
    int local_id_ = 0;

    // Externally defined identifier. For example a person's name, database reference or URL. Assumed to be long-lived.
    std::string external_id_;

    // Where is the person's face in the current view
    dlib::rectangle bounding_box_;

    // Image of the person's face as last seen
    Image face_image_;

    // Measure of the amount of blurring in the current face image
    double face_blur_;

    int non_visible_frames_ = 0;

    // Face descriptor used to determine if two faces are the same
    FaceDescriptor face_descriptor_;
};


// Manages a list of tracked objects
class Manager {
public:
    Manager(FaceDetector &face_detector) : face_detector_(face_detector) {
    }

    /*
     * Tell the manager about a new frame in which motion has been detected, or which
     * otherwise should be processed. The managed will update its knowledge of the world
     * based on the contents of this image.
     */
    void newFrame(int frame_no, cv::Mat &frame);

    std::vector<std::shared_ptr<Person>> visiblePeople() const;

    int visibleCount() const;

    int knownCount() const;

    bool isSamePerson(const FaceDescriptor &face1, const FaceDescriptor &face2) const;

    bool isSameRegion(const dlib::rectangle &bb1, const dlib::rectangle &bb2) const;

    /*
     * Tell the manager about a new person and provide a file the face can be loaded from.
     * The supplied image must contain exactly one face of close to 150x150 pixels.
     * TODO Also need ability to add person using already known face descriptor and pre-loaded image
     */
    std::shared_ptr<Person> addPerson(const std::string &external_id, const std::string &face_filename);

    // Find a person using a descriptor. Returns nullptr if no face found
    std::shared_ptr<Person> findPerson(const FaceDescriptor &descriptor) const;

    /*
     * Find a person using a bounding box
     * Returns a list since there may be multiple people overlapping the bounding box
     */
    std::vector<std::shared_ptr<Person>> findPerson(dlib::rectangle &bounding_box) const;

    /*
     * find a person using the external ID.
     * Can't guarantee that external names are unique since we don't control them so may return multiple matches.
     */
    std::vector<std::shared_ptr<Person>> findPerson(std::string &external_id) const;

    // find a person using the local ID
    std::shared_ptr<Person> findPerson(int local_id) const;

    /*
     * get / set how often detectors are run
     */
    int detectorFrameInterval() const {
        return detector_frame_interval_;
    }

    void detectorFrameInterval(int interval) {
        detector_frame_interval_ = interval;
    }

    /*
     * Clear current state but not set of known people
     */
    void reset();

private:
    void personVisible(int local_id);

    void personNotVisible(int local_id);

    /*
     * Compute a face descriptor from a face rectangle. Using jitter will compute a mean
     * of multiple perturbed versions of the image (minor changes in position, rotation and left/right flip)
     * which may give better recognition results but which is slower so we probably don't
     * want to do this every time we are testing a potentially unknown face.
     */
    FaceDescriptor getFaceDescriptor(const dlib::cv_image<dlib::bgr_pixel> &image,
                                     const dlib::rectangle &face_bounds,
                                     bool use_jitter);

    std::shared_ptr<Person> handleNewPerson(dlib::cv_image<dlib::bgr_pixel> image,
                                            dlib::rectangle &rectangle);

    std::shared_ptr<Person> makePerson(const dlib::rectangle &rectangle, const Image &face_image, double blur,
                                       const FaceDescriptor &face_descriptor);

    // Handles detecting and recognising faces
    FaceDetector &face_detector_;

    // People who are known to the system. This "owns" the Person instances
    std::map<int, std::shared_ptr<Person>> people_;

    // Map local ID to the tracker currently tracking the object with this ID
    std::map<int, std::unique_ptr<dlib::correlation_tracker>> trackers_;

    int last_frame_ = 0;

    int last_local_id_ = 0;

    bool use_jitter_ = false;

    // Maximum difference between two face descriptors to treat as same person
    float descriptor_threshold_ = 0.6;

    // Minimum Intersection over Union (IoU) value to treat bounding boxes as the same
    // TODO The slower the frame rate the lower the bounding box threshold needs to be as faces could have moved further between frames
    float bounding_box_threshold_ = 0.5;

    double min_tracker_confidence_ = 7;

    // Margins around object (face) to use when instantiating a new tracker
    int tracker_horizontal_margin_ = 10;
    int tracker_vertical_margin_ = 20;

    // number of frames between each run of the face detector. 1 means every frame
    int detector_frame_interval_ = 5;
};

#endif //FINAL_PROJECT_PERSON_H
