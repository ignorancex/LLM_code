#include "test_utils.h"
#include "mtcnn/detector.h"

#include <filesystem>

namespace fs = std::filesystem;

namespace {
cv::Mat downscaleImageToWidth(const cv::Mat& inputImage, int targetWidth) {
    // Get the original dimensions
    int originalWidth = inputImage.cols;
    int originalHeight = inputImage.rows;

    if (originalWidth < targetWidth)
        return inputImage;

    // Calculate the scaling factor
    double scaleFactor = static_cast<double>(targetWidth) / originalWidth;

    // Calculate the new height while maintaining the aspect ratio
    int targetHeight = static_cast<int>(originalHeight * scaleFactor);

    // Resize the image
    cv::Mat outputImage;
    cv::resize(inputImage, outputImage, cv::Size(targetWidth, targetHeight));

    return outputImage;
}
} // namespace

std::string getEmotiEffLibRootDir() {
    const char* emotiEffLibRoot = std::getenv("EMOTIEFFLIB_ROOT");
    if (emotiEffLibRoot == nullptr)
        throw std::runtime_error(
            "EMOTIEFFLIB_ROOT environment variable MUST be specified for running tests");
    return std::string(emotiEffLibRoot);
}

std::vector<cv::Mat> recognizeFaces(const cv::Mat& frame, int downscaleWidth) {
    fs::path dirWithModels(getEmotiEffLibRootDir());
    dirWithModels =
        dirWithModels / "emotieffcpplib" / "3rdparty" / "opencv-mtcnn" / "data" / "models";
    ProposalNetwork::Config pConfig;
    pConfig.protoText = dirWithModels / "det1.prototxt";
    pConfig.caffeModel = dirWithModels / "det1.caffemodel";
    pConfig.threshold = 0.6f;
    RefineNetwork::Config rConfig;
    rConfig.protoText = dirWithModels / "det2.prototxt";
    rConfig.caffeModel = dirWithModels / "det2.caffemodel";
    rConfig.threshold = 0.7f;
    OutputNetwork::Config oConfig;
    oConfig.protoText = dirWithModels / "det3.prototxt";
    oConfig.caffeModel = dirWithModels / "det3.caffemodel";
    oConfig.threshold = 0.7f;
    MTCNNDetector detector(pConfig, rConfig, oConfig);
    auto scaledFrame = downscaleImageToWidth(frame, downscaleWidth);
    double downcastRatioW = static_cast<double>(frame.cols) / scaledFrame.cols;
    double downcastRatioH = static_cast<double>(frame.rows) / scaledFrame.rows;
    std::vector<Face> faces = detector.detect(scaledFrame, 20.f, 0.709f);
    std::vector<cv::Mat> cvFaces;
    cvFaces.reserve(faces.size());
    for (auto& face : faces) {
        face.bbox.x1 *= downcastRatioW;
        face.bbox.x2 *= downcastRatioW;
        face.bbox.y1 *= downcastRatioH;
        face.bbox.y2 *= downcastRatioH;
        cv::Rect roi(face.bbox.x1, face.bbox.y1, face.bbox.x2 - face.bbox.x1,
                     face.bbox.y2 - face.bbox.y1);
        cv::Mat f = frame(roi).clone();
        cvFaces.push_back(f);
    }
    return cvFaces;
}

std::string getPathToPythonTestDir() {
    fs::path testDir(getEmotiEffLibRootDir());
    testDir /= "tests";

    return testDir;
}
