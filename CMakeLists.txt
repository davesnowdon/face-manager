cmake_minimum_required(VERSION 2.8.12)

PROJECT(face-manager)

#MAC USERS UNCOMMENT THE LINE BELOW if cmake gives error of not able to find OpencvConfig.cmake
#SET(OpenCV_DIR /usr/local/Cellar/opencv3/3.2.0/share/OpenCV/)

#LINUX USERS UNCOMMENT THE LINE BELOW if cmake gives error of not able to find OpencvConfig.cmake
#SET(OpenCV_DIR /usr/local/share/OpenCV/)

find_package( OpenCV REQUIRED )
find_package( dlib REQUIRED )

# TODO Fix complaints about C++11 support not enabled when built
#ADD_LIBRARY(manager STATIC motiondetector.cpp imagelogger.cpp mkpath.c manager.cpp manager.h facedetector.cpp facedetector.h)

#ADD_EXECUTABLE(manager-benchmark manager-benchmark.cpp)
#TARGET_LINK_LIBRARIES(manager-benchmark manager ${OpenCV_LIBS} dlib::dlib)

ADD_EXECUTABLE(manager-benchmark manager-benchmark.cpp motiondetector.cpp imagelogger.cpp mkpath.c manager.cpp manager.h facedetector.cpp facedetector.h demo-util.cpp demo-util.h util.h)
TARGET_LINK_LIBRARIES(manager-benchmark ${OpenCV_LIBS} dlib::dlib)

ADD_EXECUTABLE(manager-demo manager-demo.cpp motiondetector.cpp imagelogger.cpp mkpath.c manager.cpp manager.h facedetector.cpp facedetector.h demo-util.cpp demo-util.h util.h)
TARGET_LINK_LIBRARIES(manager-demo ${OpenCV_LIBS} dlib::dlib)

ADD_EXECUTABLE(micro-benchmarks micro-benchmarks.cpp)
TARGET_LINK_LIBRARIES(micro-benchmarks ${OpenCV_LIBS} dlib::dlib)
