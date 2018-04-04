# Face tracker

This code is designed to monitor a video stream by being fed individual frames and track all the people present.

The tracker will attempt to recognise individuals so if a person leaves the camera view and then returns
they should be recognised as having been seen before even if their identity is not
known.

## Demo
This accepts an input video and a list of known people and produces an output video
annotated with the current frame rate (calculated using an exponential moving average),
the faces currently being tracked and their identities if known.

    ./manager-demo test-data/movinghuman1.mp4 diff Dave face1.jpg Devlin face2.jpg


## Benchmarks

### Manager benchmarks
These are intended to compare the various motion detection methods and get a feel for the
performance improvements when using the manager compared to a naive implementation.

    ./manager-benchmark test-data/movinghuman2.mp4 1

### Micro benchmarks
These are intended to get rough performance figures for the basic operations performed
by the face tracking and motion detection code. The aim is to guide the implementation and
get a feel for how expensive the various operations are on a desktop and Raspberry Pi

    ./micro-benchmarks test-data/example-frame.png

## Tests
Tests use [Catch2](https://github.com/catchorg/Catch2)

## Code style
Generally follows [Google C++ style](https://google.github.io/styleguide/cppguide.html) with the following exceptions:
* constants use Java style ALL_CAPS.
* indentation is 4 spaces not 2

Feel free to point out places where I need to fix code style :-) 

## License
This code is licensed under the [Boost 1.0 license](http://www.boost.org/users/license.html)

## Thanks
* Many thanks to [Davis King](https://github.com/davisking) ([@nulhom](https://twitter.com/nulhom)) for creating [dlib](http://dlib.net/) and for providing the trained facial feature detection and face encoding models used in this library. For more information on the ResNet that powers the face encodings, check out his [blog post](http://blog.dlib.net/2017/02/high-quality-face-recognition-with-deep.html).
