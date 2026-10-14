# DOPE + ViSP 3D 6DoF Tracking Pipeline

This document describes the pipeline integrating NVIDIA's **DOPE (Deep Object Pose Estimation)** for automatic initialization with **ViSP (Visual Servoing Platform)** for tracking.

The buildable implementation is [object_tracking.cpp](./object_tracking.cpp). The
older illustrative code below uses ViSP calls that do not match the installed
ViSP API; build and run the linked source rather than copying that example.

## Reference C++ Example

```cpp
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <visp3/core/vpHomogeneousMatrix.h>
#include <visp3/mbt/vpMbEdgeKltTracker.h>
#include <visp3/core/vpImageConvert.h>
#include <iostream>
#include <vector>

// Global flag to manage the state machine
bool g_is_tracking = false;

// 1. Precise 3D local coordinates of the YCB Object Cuboid (Example: Mustard Bottle)
// Order must match DOPE's internal output scheme (usually 8 corners + 1 centroid)
std::vector<cv::Point3f> getYCBOuboid3DPoints() {
    std::vector<cv::Point3f> modelPoints;
    // Centroid
    modelPoints.push_back(cv::Point3f(0.0f, 0.0f, 0.0f));
    // 8 Cuboid corners (Dimensions in meters to match ViSP environment)
    float minX = -0.0480121f, maxX = 0.0480121f; // width
    float minY = -0.0956505f, maxY = 0.0956505f; // height
    float minZ = -0.0291245f, maxZ = 0.0291245f; // depth

    modelPoints.push_back(cv::Point3f(minX, minY, minZ));
    modelPoints.push_back(cv::Point3f(maxX, minY, minZ));
    modelPoints.push_back(cv::Point3f(maxX, maxY, minZ));
    modelPoints.push_back(cv::Point3f(minX, maxY, minZ));
    modelPoints.push_back(cv::Point3f(minX, minY, maxZ));
    modelPoints.push_back(cv::Point3f(maxX, minY, maxZ));
    modelPoints.push_back(cv::Point3f(maxX, maxY, maxZ));
    modelPoints.push_back(cv::Point3f(minX, maxY, maxZ));
    return modelPoints;
}

// 2. Core function to run DOPE ONNX model and calculate initial 6DoF
bool runDopePoseInitializer(cv::dnn::Net& net, const cv::Mat& frame, cv::Mat& rvec, cv::Mat& tvec, const cv::Mat& camMatrix, const cv::Mat& distCoeffs) {
    // DOPE typical input: 448x448 resolution normalized
    cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0 / 255.0, cv::Size(448, 448), cv::Scalar(0,0,0), true, false);
    net.setInput(blob);

    std::vector<cv::Mat> outputs;
    net.forward(outputs); // Architecture evaluates belief maps

    if (outputs.empty()) return false;

    // Output parsing: Identify top peak location for each of the 9 keypoint feature maps
    cv::Mat beliefMaps = outputs[0]; 
    std::vector<cv::Point2f> imagePoints2D;

    int num_points = 9;
    for (int i = 0; i < num_points; ++i) {
        // Slice the tensor map for point 'i'
        cv::Mat scoreMap(56, 56, CV_32F, beliefMaps.ptr<float>(0, i));
        
        double maxVal;
        cv::Point maxLoc;
        cv::minMaxLoc(scoreMap, nullptr, &maxVal, nullptr, &maxLoc);

        // Filter false detections/low confidence threshold
        if (maxVal < 0.5) return false; 

        // Scale point location back up from tensor grid (56x56) to actual image coordinates
        float scaleX = (float)frame.cols / 56.0f;
        float scaleY = (float)frame.rows / 56.0f;
        imagePoints2D.push_back(cv::Point2f(maxLoc.x * scaleX, maxLoc.y * scaleY));
    }

    // Solve PnP RANSAC using the extracted points vs 3D geometry
    std::vector<cv::Point3f> objectPoints3D = getYCBOuboid3DPoints();
    bool pnp_success = cv::solvePnPRansac(objectPoints3D, imagePoints2D, camMatrix, distCoeffs, rvec, tvec);
    
    return pnp_success;
}

// 3. Helper to build ViSP compatible translation and rotation frame
vpHomogeneousMatrix buildViSPMatrix(const cv::Mat& rvec, const cv::Mat& tvec) {
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    vpHomogeneousMatrix cMo;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) cMo[i][j] = R.at<double>(i, j);
        cMo[i] = tvec.at<double>(i); // In meters
    }
    return cMo;
}

int main() {
    // Initialize Camera Intrinsic Parameters Matrix
    cv::Mat cameraMatrix = (cv::Mat_<double>(3,3) << 615.0, 0, 320.0, 0, 615.0, 240.0, 0, 0, 1);
    cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64F);

    // Initialize OpenCV DNN with DOPE ONNX binary
    cv::dnn::Net dopeNet = cv::dnn::readNetFromONNX("dope_mustard_bottle.onnx");
    dopeNet.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    dopeNet.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    // Initialize ViSP Tracker Instance
    vpImage<unsigned char> vispFrame;
    vpMbEdgeKltTracker tracker;
    tracker.loadConfigFile("mustard.xml");
    tracker.loadModel("mustard.cao");

    cv::VideoCapture cap(0); // Open WebCam / RealSense Stream

    while (cap.isOpened()) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        vpImageConvert::convert(frame, vispFrame);

        if (!g_is_tracking) {
            // STEP A: RUN DOPE INITIALIZER (CPU/GPU bound single execution)
            cv::Mat rvec, tvec;
            if (runDopePoseInitializer(dopeNet, frame, rvec, tvec, cameraMatrix, distCoeffs)) {
                
                vpHomogeneousMatrix initial_pose = buildViSPMatrix(rvec, tvec);
                
                // Hand over initial coordinates matrix seamlessly to ViSP framework
                tracker.initFromPose(vispFrame, initial_pose);
                g_is_tracking = true;
                std::cout << "🚀 DOPE accurately initialized ViSP tracking engine!" << std::endl;
            }
        } 
        else {
            // STEP B: HIGH-SPEED VISP TRACKING LAYER (60+ FPS)
            try {
                tracker.track(vispFrame);
                vpHomogeneousMatrix trackedPose = tracker.getPose();
                
                // Track quality control
                if (tracker.getConfidence() < 0.45) {
                    std::cout << "❌ Tracking lost confidence. Falling back to DOPE..." << std::endl;
                    g_is_tracking = false;
                } else {
                    // Display or stream your exact 6DoF Pose downstream to an actor/robot link
                    trackedPose.print(); 
                }
            } 
            catch (...) {
                g_is_tracking = false; // Graceful structural exception reset
            }
        }
    }
    return 0;
}
```

## Local Windows Build and Run

The workspace uses ViSP 3.7.0 built under `dependencies\visp-install` and the
OpenCV 4.10 SDK already installed on this machine. Configure and build from
PowerShell with:

```powershell
cmake -S . -B build -A x64 `
  -DOpenCV_DIR="C:\dev\opencv_camera_20260626_2_opencv_prebuilt\opencv_prebuilt\build\x64\vc16\lib" `
  -DVISP_DIR="$PWD\dependencies\visp-install\x64\vc18\lib"
cmake --build build --config Release --parallel 8
```

Run the tracker with the installed model and a camera calibrated for the
intrinsics in `dependencies\models\mustard.xml`:

```powershell
.\build\Release\object_tracking.exe `
  .\dependencies\models\dope_mustard_bottle.onnx `
  .\dependencies\models\mustard.xml `
  .\dependencies\models\mustard.cao 0
```

On Windows, `build.bat` configures and builds the Release executable, and
`run.bat` launches it with camera index 0. Pass a different camera index as an
optional argument, for example `run.bat 1`. If the OpenCV or ViSP installation
is in a different location, set `OpenCV_DIR` or `VISP_DIR` before running
`build.bat`.

The official NVIDIA `Mustard.pth` checkpoint and the YCB Mustard Bottle mesh
are in `dependencies\models`. Re-export the ONNX file with the project venv and
`tools\export_dope_onnx.py`; regenerate the ViSP convex-hull model with
`tools\convert_ycb_obj_to_cao.py`. Python conversion packages are listed in
`dependencies\dope-requirements.txt`.

The DOPE source and checkpoint are covered by NVIDIA's Source Code License,
which limits use to non-commercial research or evaluation; review
`dependencies\dope-source\LICENSE.md` before use or redistribution. YCB mesh
provenance and its CC BY 4.0 terms are recorded in
`dependencies\models\ycb\SOURCE.md`.
