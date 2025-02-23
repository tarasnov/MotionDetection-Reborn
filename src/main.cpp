#include <opencv2/opencv.hpp>
#include <iostream>

#include "include/motion_detection.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path-to-video>" << std::endl;
        return 1;
    }

    // Get the video file path from the command line.
    std::string video_path = argv[1];
    std::cout << "Opening video: " << video_path << std::endl;

    // Open the default camera (or use a video file by providing the filename)
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video stream." << std::endl;
        return -1;
    }

    MovementDetection detector("dummy_path", std::chrono::seconds(1));
    
    for (int i = 0; i < 120; i++) {
        static cv::Mat frame;
        cap >> frame;  // Capture a new frame
        if (frame.empty()) break;  // End of video stream

        // Process the frame (this is where your algorithm goes)
        cv::imshow("Frame", frame);
        if (cv::waitKey(30) >= 0) break;

        detector.PushFrame(frame, std::chrono::steady_clock::now(), false);
    }
}
