cmake_minimum_required(VERSION 3.10)
project(MovementDetectionCPP)

# Set additional optimization flags for Release builds.
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -flto")

# Find dependencies
find_package(OpenCV REQUIRED)
find_package(HailoRT REQUIRED)

# Fetch GoogleTest (which includes GoogleMock)
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/release-1.12.1.tar.gz
)
FetchContent_MakeAvailable(googletest)

# Include directories for our project
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/include)

# Create a library from your source files (excluding main.cpp and test files)
add_library(movementDetectionLib
        src/motion_detection.cpp
        src/feature_extractor.cpp
        src/keypoint_matcher.cpp
        src/model.cpp
)
target_link_libraries(movementDetectionLib PRIVATE ${OpenCV_LIBS} HailoRT::libhailort)

# Build the main executable from main.cpp
add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE movementDetectionLib)

# Build the test executable from tests/test_hailo.cpp
add_executable(MyTests tests/test_hailo.cpp)
target_link_libraries(MyTests PRIVATE movementDetectionLib gtest gtest_main)

# Discover and add tests to CTest
include(GoogleTest)
gtest_discover_tests(MyTests)

# Optional: if you decide to use pybind11, add and link it here:
# add_subdirectory(pybind11)
# target_link_libraries(${PROJECT_NAME} PRIVATE pybind11::embed)
