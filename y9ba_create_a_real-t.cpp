#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/aruco.hpp>

using namespace cv;
using namespace std;

#define MODULE_TRACKING_THRESHOLD 0.5 // adjust this value to fine-tune tracking accuracy

struct ModuleTracker {
    Mat moduleImage;
    vector<Point2f> moduleCorners;
    vector<int> moduleIds;
};

vector<ModuleTracker> moduleTrackers;

void detectModules(Mat frame) {
    vector<vector<Point2f>> corners;
    vector<vector<int>> ids;
    aruco::detectMarkers(frame, markerDict, corners, ids);
    
    for (int i = 0; i < ids.size(); i++) {
        ModuleTracker tracker;
        tracker.moduleIds = ids[i];
        tracker.moduleCorners = corners[i];
        tracker.moduleImage = frame.clone();
        drawContours(tracker.moduleImage, corners, -1, Scalar(0, 255, 0), 2);
        moduleTrackers.push_back(tracker);
    }
}

void trackModules(Mat frame) {
    for (ModuleTracker& tracker : moduleTrackers) {
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        vector<Point2f> newCorners;
        vector<int> newIds;
        
        for (int i = 0; i < tracker.moduleIds.size(); i++) {
            Point2f corner = tracker.moduleCorners[i];
            Point2f newCorner = corner + Point2f(1, 1);
            float response = 0;
            cornerSubPix(gray, corner, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.01));
            cornerSubPix(gray, newCorner, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.01));
            response = getOpticalFlowPyrLK(gray, frame, corner, newCorner, Size(11, 11), 5, TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.01));
            
            if (response > MODULE_TRACKING_THRESHOLD) {
                newCorners.push_back(newCorner);
                newIds.push_back(tracker.moduleIds[i]);
            }
        }
        
        tracker.moduleCorners = newCorners;
        tracker.moduleIds = newIds;
        tracker.moduleImage = frame.clone();
        drawContours(tracker.moduleImage, newCorners, -1, Scalar(0, 255, 0), 2);
    }
}

int main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        return -1;
    }
    
    Ptr<aruco::Dictionary> markerDict = aruco::getPredefinedDictionary(aruco::DICT_ARUCO_ORIGINAL);
    
    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) {
            break;
        }
        
        detectModules(frame);
        trackModules(frame);
        
        for (ModuleTracker tracker : moduleTrackers) {
            imshow("Module Tracker", tracker.moduleImage);
        }
        
        if (waitKey(1) >= 0) {
            break;
        }
    }
    
    return 0;
}