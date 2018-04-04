/*
 *  Face tracker 0.1
 *
 *  Copyright (c) 2018 David Snowdon. All rights reserved.
 *
 *  Distributed under the Boost Software License, Version 1.0. (See accompanying
 *  file LICENSE.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */


#include "manager.h"
#include "imagelogger.h"
#include "util.h"
#include <algorithm>

#include <dlib/image_io.h>

bool
rectangleComparator(const dlib::rectangle &l, const dlib::rectangle &r) {
    return l.left() < r.left();
}

bool
personComparator(const std::shared_ptr<Person> &l, const std::shared_ptr<Person> &r) {
    return rectangleComparator(l->boundingBox(), r->boundingBox());
}

void
Manager::sortActive() {
    std::sort(visible_people_.begin(), visible_people_.end(), personComparator);
}

void
Manager::newFrame(int frame_no, cv::Mat &frame) {
    dlib::cv_image<dlib::bgr_pixel> frame_dlib(frame);

    // Update the trackers
    std::vector<int> low_confidence_trackers;
    for (auto it = trackers_.begin(); it != trackers_.end(); ++it) {
        double confidence = it->second->update(frame_dlib);
        if (logger.debugEnabled()) {
            logger.debug(
                    "Tracker for : " + std::to_string(it->first) + " has confidence " + std::to_string(confidence));
        }

        if (confidence < min_tracker_confidence_) {
            low_confidence_trackers.push_back(it->first);
        }
    }

    if (low_confidence_trackers.size() > 0) {
        if (logger.debugEnabled()) {
            logger.debug(std::to_string(low_confidence_trackers.size()) + " trackers with confidence less than " +
                         std::to_string(min_tracker_confidence_) + " to dispose of");
        }
        for (auto it = low_confidence_trackers.begin(); it != low_confidence_trackers.end(); ++it) {
            trackers_.erase(*it);
        }
    }

    // Detect faces in the image
    // TODO Would it be useful to make this adaptive based on frame rate?
    if (0 == (frame_no % detector_frame_interval_)) {
        std::vector<dlib::rectangle> faceRects = face_detector_.detectFaces(frame_dlib);
        if (logger.debugEnabled()) {
            logger.debug("Number of faces detected: " +
                         std::to_string(faceRects.size()) +
                         ", current visible faces: " +
                         std::to_string(visible_people_.size()));
        }

        // which local IDs have been matched with detected faces
        std::set<int> matched_ids;

        if (faceRects.size() > 0) {
            for (auto itf = faceRects.begin(); itf != faceRects.end(); ++itf) {
                dlib::rectangle &face_rect = *itf;
                if (logger.debugEnabled()) {
                    logger.debug("Face rectangle (from detector): ", face_rect);
                }

                // face centre from detector
                long face_centre_x = face_rect.left() + face_rect.width() / 2;
                long face_centre_y = face_rect.top() + face_rect.height() / 2;

                /*
                 * Now compare the detected faces with the tracked faces
                 *
                 * TODO Sorting the tracker and face rectangles might allow us to avoid the O(N^2) loop but
                 * this is unlikely to be worth the effort unless we are tracking large numbers of objects
                 */
                bool is_face_matched = false;
                int matched_id = 0;
                for (auto itt = trackers_.begin(); itt != trackers_.end(); ++itt) {
                    int tracker_local_id = itt->first;
                    dlib::rectangle tracker_rect = itt->second->get_position();
                    if (logger.debugEnabled()) {
                        logger.debug("Face rectangle (from tracker): ", tracker_rect);
                    }

                    // face centre from tracker
                    long tracker_centre_x = tracker_rect.left() + tracker_rect.width() / 2;
                    long tracker_centre_y = tracker_rect.top() + tracker_rect.height() / 2;

                    /*
                     * Determine if this tracker matches the detected face
                     *
                     * Check if the centre of the face is within the  rectangle of the tracker region.
                     * Also, the centre of the tracker region must be within the region  detected as a face.
                     * If both of these conditions hold we have a match.
                     *
                     * TODO Would IoU be a better measure?
                     * TODO handler multiple overlapping rectangles, find best match
                     */
                    if ((tracker_rect.left() <= face_centre_x) &&
                        (face_centre_x <= tracker_rect.right()) &&
                        (tracker_rect.top() <= face_centre_y) &&
                        (face_centre_y <= tracker_rect.bottom()) &&
                        (face_rect.left() <= tracker_centre_x) &&
                        (tracker_centre_x <= face_rect.right()) &&
                        (face_rect.top() <= tracker_centre_y) &&
                        (tracker_centre_y <= face_rect.bottom())) {

                        if (!is_face_matched) {
                            logger.debug("Detected face and tracked face match. Local ID = " +
                                         std::to_string(tracker_local_id));
                            is_face_matched = true;
                            matched_id = tracker_local_id;
                        } else {
                            logger.debug("Duplicate tracker/face match Local IDs = " +
                                         std::to_string(matched_id) + " & " +
                                         std::to_string(tracker_local_id));
                            // TODO handle duplicates by finding best match
                        }
                        matched_ids.insert(tracker_local_id);
                    }
                }

                /*
                 * Did we detect a new face? This could be a face we've seen before but has been off camera so
                 * we need to calculate a face descriptor and compare with descriptors we've see before.
                 */
                if (!is_face_matched) {
                    logger.debug("New face detected at ", face_rect);
                    FaceDescriptor descriptor = getFaceDescriptor(frame_dlib, face_rect);
                    auto known_person = findPerson(descriptor);
                    int new_tracker_id = 0;
                    if (!known_person) {
                        // Person we have not seen before
                        new_tracker_id = ++last_local_id_;

                        // TODO set up new person
                    } else {
                        // Person we've seen before
                        new_tracker_id = known_person->localId();
                    }

                    dlib::rectangle padded_rectangle(face_rect.left() - tracker_horizontal_margin_,
                                                     face_rect.top() - tracker_vertical_margin_,
                                                     face_rect.right() + tracker_horizontal_margin_,
                                                     face_rect.bottom() + tracker_vertical_margin_);
                    dlib::correlation_tracker *tracker = new dlib::correlation_tracker();
                    tracker->start_track(frame_dlib, padded_rectangle);
                    trackers_[new_tracker_id] = std::unique_ptr<dlib::correlation_tracker>(tracker);
                    if (logger.debugEnabled()) {
                        logger.debug("New tracker for " + std::to_string(new_tracker_id), padded_rectangle);
                    }
                }
            }
        }

        // now we need to handle any leftover trackers that were not matched up with faces
        // set of all tracked local IDs
        std::set<int> tracked_ids = extract_keys(trackers_);
        std::set<int> difference;
        std::set_difference(tracked_ids.begin(), tracked_ids.end(),
                            matched_ids.begin(), matched_ids.end(),
                            std::inserter(difference, difference.begin()));
        if (logger.debugEnabled()) {
            logger.debug("Found " + std::to_string(difference.size()) +
                         " local IDS that are tracked but not detected: " +
                         set_to_string(difference, ","));
        }
        for (int id : difference) {
            trackers_.erase(id);
        }
    }
}

bool
Manager::isSamePerson(const FaceDescriptor &face1, const FaceDescriptor &face2) const {
    return dlib::length(face1 - face2) < descriptor_threshold_;
}

bool
Manager::isSameRegion(const dlib::rectangle &bb1, const dlib::rectangle &bb2) const {
    return dlib::box_intersection_over_union(bb1, bb2) > bounding_box_threshold_;
}

// Find a person using a descriptor. Returns nullptr if no face found
std::shared_ptr<Person>
Manager::findPerson(const FaceDescriptor &descriptor) const {
    for (const std::shared_ptr<Person> &person : seen_people_) {
        if (isSamePerson(descriptor, person->faceDescriptor())) {
            return person;
        }
    }
    return nullptr;
}

std::shared_ptr<Person>
Manager::addPerson(const std::string &external_id, const std::string &face_filename) {

    // load image from file
    dlib::array2d<dlib::rgb_pixel> img;
    dlib::load_image(img, face_filename);

    // check that a single face can be detected
    std::vector<dlib::rectangle> face_bbs = face_detector_.detectFaces(img);
    if (1 != face_bbs.size()) {
        // TODO raise exception
        return nullptr;
    }

    // extract face image
    dlib::rectangle face_box = face_bbs[0];
    dlib::matrix<dlib::rgb_pixel> face_chip = face_detector_.extractFaceImage(img, face_box);

    // get face descriptor - use jitter to make the descriptor more resistant to noise
    FaceDescriptor descriptor = face_detector_.getFaceDescriptor(face_chip, true);

    Image tmp_face_image; // convert from matrix to array2d
    dlib::assign_image(tmp_face_image, face_chip);

    // create person and store details in seen list
    // TODO calculate blur
    double blur = 0;
    // TODO does not make a lot of sense to include the bounding box
    auto person = makePerson(face_box, tmp_face_image, blur, descriptor);
    person->externalId(external_id);

    // Remember the person so we can identify them if seen
    seen_people_.push_back(person);
    return person;
}


/*
 * Find a person using a bounding box. Only checks for people that
 * are currently visible.
 */
std::shared_ptr<Person>
Manager::findPerson(dlib::rectangle &bounding_box) const {
    for (const std::shared_ptr<Person> &person : seen_people_) {
        if (isSameRegion(bounding_box, person->boundingBox())) {
            return person;
        }
    }
    return nullptr;
}

// find a person using the external ID
std::shared_ptr<Person>
Manager::findPerson(std::string &external_id) const {
    for (const std::shared_ptr<Person> &person : seen_people_) {
        if (external_id == person->externalId()) {
            return person;
        }
    }
    return nullptr;
}

// find a person using the local ID
std::shared_ptr<Person>
Manager::findPerson(int local_id) const {
    for (const std::shared_ptr<Person> &person : seen_people_) {
        if (local_id == person->localId()) {
            return person;
        }
    }
    return nullptr;
}

void
Manager::handleNoFacesDetected() {
    // no faces detected increment all non-visible counts and check thresholds
    for (const std::shared_ptr<Person> &person : visible_people_) {
        person->incNonVisibleFrames();
    }
}


FaceDescriptor
Manager::getFaceDescriptor(const dlib::cv_image<dlib::bgr_pixel> &image,
                           const dlib::rectangle &face_bounds) {
    // These are the transformed and extracted faces
    dlib::matrix<dlib::rgb_pixel> face = face_detector_.extractFaceImage(image, face_bounds);

    /*
     * This call asks the DNN to convert each face image in faces into a 128D vector.
     * In this 128D vector space, images from the same person will be close to each other
     * but vectors from different people will be far apart.  So we can use these vectors to
     * identify if a pair of images are from the same person or from different people.
     */
    return face_detector_.getFaceDescriptor(face, use_jitter_);
}

void
Manager::handleNewFaceBox(dlib::cv_image<dlib::bgr_pixel> image, dlib::rectangle &rectangle) {
    // These are the transformed and extracted faces
    dlib::matrix<dlib::rgb_pixel> face = face_detector_.extractFaceImage(image, rectangle);

    // TODO determine amount of blurriness
    double blur = 0;

    /*
     * This call asks the DNN to convert each face image in faces into a 128D vector.
     * In this 128D vector space, images from the same person will be close to each other
     * but vectors from different people will be far apart.  So we can use these vectors to
     * identify if a pair of images are from the same person or from different people.
     */
    FaceDescriptor face_descriptor = face_detector_.getFaceDescriptor(face, use_jitter_);
    Image tmp_face_image; // convert from matrix to array2d
    dlib::assign_image(tmp_face_image, face);

    // Put person on known list and currently visible list
    auto person = makePerson(rectangle, tmp_face_image, blur, face_descriptor);
    seen_people_.push_back(person);
    visible_people_.push_back(person);
}

std::shared_ptr<Person>
Manager::makePerson(const dlib::rectangle &rectangle, const Image &face_image, double blur,
                    const FaceDescriptor &face_descriptor) {
    return std::make_shared<Person>(++last_local_id_, rectangle, face_image, blur, face_descriptor);
}

void
Manager::purgeVisibleList() {
    // Remove any people that have not been visible for at least max_non_visible_frames_ frames
    int prev_visible = visible_people_.size();
    visible_people_.erase(
            std::remove_if(visible_people_.begin(), visible_people_.end(),
                           [this](const std::shared_ptr<Person> &person) {
                               return person->nonVisibleFrames() >= max_non_visible_frames_;
                           }),
            visible_people_.end());

    if (logger.debugEnabled()) {
        logger.debug("# people removed from visible list: " +
                     std::to_string(prev_visible - visible_people_.size()));
    }
}

void Manager::reset() {
    last_frame_ = 0;
    last_local_id_ = 0;
    visible_people_.clear();
}
