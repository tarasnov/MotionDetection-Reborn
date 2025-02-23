cmake_minimum_required(VERSION 3.10)
project(MovementDetectionCPP)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Optionally, if you decide to use pybind11 for NumPy support, you can add it here:
# add_subdirectory(pybind11)  # Assuming you clone pybind11 into your project

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# List your source files
set(SOURCES
    src/main.cpp
    src/motion_detection.cpp
    src/feature_extractor.cpp
    src/keypoint_matcher.cpp
    src/model.cpp
)

# Optionally, list headers so that IDEs recognize them
set(HEADERS
        src/include/motion_detection.h
        src/include/feature_extractor.h
        src/include/keypoint_matcher.h
        src/include/model.h
)

# Add the executable
add_executable(MovementDetectionCPP ${SOURCES} ${HEADERS})

# Link against OpenCV libraries
target_link_libraries(MovementDetectionCPP ${OpenCV_LIBS})

# If using pybind11, link it as well:
# target_link_libraries(MovementDetectionCPP PRIVATE pybind11::embed)
