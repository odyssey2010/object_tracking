#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <visp3/core/vpCameraParameters.h>
#include <visp3/core/vpException.h>
#include <visp3/core/vpHomogeneousMatrix.h>
#include <visp3/core/vpImage.h>
#include <visp3/core/vpImageConvert.h>
#include <visp3/mbt/vpMbEdgeKltTracker.h>

#include <exception>
#include <iostream>
#include <string>
#include <vector>

#ifdef ENABLE_VISP_NAMESPACE
using namespace VISP_NAMESPACE_NAME;
#endif

namespace
{
constexpr int kInputSize = 448;
constexpr int kExpectedKeypoints = 9;
constexpr float kKeypointThreshold = 0.5f;

std::vector<cv::Point3f> getYcbCuboidPoints()
{
  constexpr float minX = -0.0480121f;
  constexpr float maxX = 0.0480121f;
  constexpr float minY = -0.0956505f;
  constexpr float maxY = 0.0956505f;
  constexpr float minZ = -0.0291245f;
  constexpr float maxZ = 0.0291245f;

  return {
    { 0.0f, 0.0f, 0.0f },
    { minX, minY, minZ },
    { maxX, minY, minZ },
    { maxX, maxY, minZ },
    { minX, maxY, minZ },
    { minX, minY, maxZ },
    { maxX, minY, maxZ },
    { maxX, maxY, maxZ },
    { minX, maxY, maxZ }
  };
}

bool getBeliefMap(const cv::Mat &output, int keypointIndex, cv::Mat &map)
{
  if (output.dims != 4 || output.size[0] != 1 || output.depth() != CV_32F) {
    return false;
  }

  if (output.size[1] == kExpectedKeypoints) {
    const int height = output.size[2];
    const int width = output.size[3];
    map = cv::Mat(height, width, CV_32F,
                  const_cast<float *>(output.ptr<float>(0, keypointIndex))).clone();
    return true;
  }

  if (output.size[3] == kExpectedKeypoints) {
    const int height = output.size[1];
    const int width = output.size[2];
    map.create(height, width, CV_32F);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        map.at<float>(y, x) = output.ptr<float>(0, y, x)[keypointIndex];
      }
    }
    return true;
  }

  if (output.size[1] > kExpectedKeypoints &&
      output.size[2] != kExpectedKeypoints && output.size[3] != kExpectedKeypoints) {
    const int height = output.size[2];
    const int width = output.size[3];
    map = cv::Mat(height, width, CV_32F,
                  const_cast<float *>(output.ptr<float>(0, keypointIndex))).clone();
    return true;
  }

  return false;
}

bool runDopePoseInitializer(cv::dnn::Net &net, const cv::Mat &frame,
                            const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs,
                            cv::Mat &rvec, cv::Mat &tvec)
{
  const cv::Mat blob = cv::dnn::blobFromImage(
    frame, 1.0 / 255.0, cv::Size(kInputSize, kInputSize), cv::Scalar(), true, false);
  net.setInput(blob);

  std::vector<cv::Mat> outputs;
  net.forward(outputs);
  if (outputs.empty()) {
    std::cerr << "DOPE returned no output tensors." << std::endl;
    return false;
  }

  const auto objectPoints = getYcbCuboidPoints();
  std::vector<cv::Point3f> validObjectPoints;
  std::vector<cv::Point2f> imagePoints;

  for (int i = 0; i < kExpectedKeypoints; ++i) {
    cv::Mat beliefMap;
    if (!getBeliefMap(outputs.front(), i, beliefMap)) {
      std::cerr << "Unsupported DOPE belief-map tensor. Expected a float NCHW or NHWC "
                   "tensor with at least nine keypoint maps."
                << std::endl;
      return false;
    }

    double maxValue = 0.0;
    cv::Point maxLocation;
    cv::minMaxLoc(beliefMap, nullptr, &maxValue, nullptr, &maxLocation);
    if (maxValue < kKeypointThreshold) {
      continue;
    }

    const float scaleX = static_cast<float>(frame.cols) / beliefMap.cols;
    const float scaleY = static_cast<float>(frame.rows) / beliefMap.rows;
    validObjectPoints.push_back(objectPoints[static_cast<std::size_t>(i)]);
    imagePoints.emplace_back(maxLocation.x * scaleX, maxLocation.y * scaleY);
  }

  if (validObjectPoints.size() < 4) {
    return false;
  }

  return cv::solvePnPRansac(validObjectPoints, imagePoints, cameraMatrix, distCoeffs,
                            rvec, tvec);
}

bool validateDopeOnnx(const std::string &modelPath)
{
  cv::dnn::Net net = cv::dnn::readNetFromONNX(modelPath);
  if (net.empty()) {
    std::cerr << "Could not load DOPE model: " << modelPath << std::endl;
    return false;
  }

  const int inputDimensions[] = { 1, 3, kInputSize, kInputSize };
  cv::Mat input(4, inputDimensions, CV_32F, cv::Scalar(0));
  net.setInput(input);
  const cv::Mat output = net.forward();

  if (output.dims != 4 || output.size[0] != 1 || output.size[1] != kExpectedKeypoints ||
      output.size[2] != 56 || output.size[3] != 56 || output.depth() != CV_32F) {
    std::cerr << "Unexpected DOPE ONNX output shape or type." << std::endl;
    return false;
  }

  std::cout << "DOPE ONNX inference passed: 1x9x56x56 float belief maps." << std::endl;
  return true;
}

vpHomogeneousMatrix toViSPPose(const cv::Mat &rvec, const cv::Mat &tvec)
{
  cv::Mat rotation;
  cv::Rodrigues(rvec, rotation);
  rotation.convertTo(rotation, CV_64F);

  cv::Mat translation;
  tvec.convertTo(translation, CV_64F);

  vpHomogeneousMatrix pose;
  for (unsigned int row = 0; row < 3; ++row) {
    for (unsigned int column = 0; column < 3; ++column) {
      pose[row][column] = rotation.at<double>(static_cast<int>(row),
                                               static_cast<int>(column));
    }
    pose[row][3] = translation.at<double>(static_cast<int>(row), 0);
  }
  return pose;
}

void printUsage(const char *program)
{
  std::cerr << "Usage: " << program
            << " <dope_model.onnx> <tracker_config.xml> <object_model.cao>"
               " [camera_index]\n"
            << "       " << program << " --validate-model <dope_model.onnx>\n";
}
} // namespace

int main(int argc, char **argv)
{
  const bool validateModel = argc == 3 && std::string(argv[1]) == "--validate-model";
  if (!validateModel && (argc < 4 || argc > 5)) {
    printUsage(argv[0]);
    return 2;
  }

  try {
    if (validateModel) {
      return validateDopeOnnx(argv[2]) ? 0 : 1;
    }

    const std::string modelPath = argv[1];
    const std::string configPath = argv[2];
    const std::string objectPath = argv[3];
    const int cameraIndex = argc == 5 ? std::stoi(argv[4]) : 0;
    cv::dnn::Net dopeNet = cv::dnn::readNetFromONNX(modelPath);
    if (dopeNet.empty()) {
      std::cerr << "Could not load DOPE model: " << modelPath << std::endl;
      return 1;
    }
#ifdef OBJECT_TRACKING_ENABLE_CUDA
    dopeNet.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    dopeNet.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
#endif

    vpMbEdgeKltTracker tracker;
    tracker.loadConfigFile(configPath);
    tracker.loadModel(objectPath);

    vpCameraParameters cameraParameters;
    tracker.getCameraParameters(cameraParameters);
    const cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) <<
      cameraParameters.get_px(), 0.0, cameraParameters.get_u0(),
      0.0, cameraParameters.get_py(), cameraParameters.get_v0(),
      0.0, 0.0, 1.0);
    const cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64F);

    cv::VideoCapture capture(cameraIndex);
    if (!capture.isOpened()) {
      std::cerr << "Could not open camera index " << cameraIndex << std::endl;
      return 1;
    }

    bool isTracking = false;
    cv::namedWindow("DOPE + ViSP tracking", cv::WINDOW_AUTOSIZE);
    while (true) {
      cv::Mat frame;
      if (!capture.read(frame) || frame.empty()) {
        std::cerr << "Camera stream ended or returned an empty frame." << std::endl;
        break;
      }

      vpImage<unsigned char> vispFrame;
      vpImageConvert::convert(frame, vispFrame);

      if (!isTracking) {
        cv::Mat rvec;
        cv::Mat tvec;
        if (runDopePoseInitializer(dopeNet, frame, cameraMatrix, distCoeffs, rvec, tvec)) {
          try {
            tracker.initFromPose(vispFrame, toViSPPose(rvec, tvec));
            isTracking = true;
            std::cout << "DOPE initialized ViSP tracking." << std::endl;
          }
          catch (const vpException &error) {
            std::cerr << "ViSP initialization failed: " << error.what() << std::endl;
          }
        }
        else {
          cv::putText(frame, "Waiting for DOPE detection", cv::Point(12, 28),
                      cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
        }
      }
      else {
        try {
          tracker.track(vispFrame);
          vpHomogeneousMatrix trackedPose;
          tracker.getPose(trackedPose);
          trackedPose.print();
        }
        catch (const vpException &error) {
          std::cerr << "ViSP tracking lost; retrying DOPE initialization: "
                    << error.what() << std::endl;
          isTracking = false;
        }
      }

      cv::imshow("DOPE + ViSP tracking", frame);
      const int key = cv::waitKey(1);
      if (key == 27 || key == 'q' || key == 'Q') {
        break;
      }
    }

    capture.release();
    cv::destroyAllWindows();
  }
  catch (const vpException &error) {
    std::cerr << "ViSP error: " << error.what() << std::endl;
    return 1;
  }
  catch (const cv::Exception &error) {
    std::cerr << "OpenCV error: " << error.what() << std::endl;
    return 1;
  }
  catch (const std::exception &error) {
    std::cerr << "Error: " << error.what() << std::endl;
    return 1;
  }

  return 0;
}
