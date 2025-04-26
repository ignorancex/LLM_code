#include "test_utils.h"
// Dirty approach to manually modify protected field
#define protected public
#include <emotiefflib/facial_analysis.h>
#undef protected
#include <filesystem>
#include <gtest/gtest.h>
#include <string>

namespace fs = std::filesystem;

class EmotiEffLibInternalTests : public ::testing::TestWithParam<std::string> {};

TEST_P(EmotiEffLibInternalTests, ImagePreprocessing) {
    std::string backend = GetParam();

    fs::path modelPath(getEmotiEffLibRootDir());
    modelPath = modelPath / "models" / "emotieffcpplib_prepared_models";
    if (backend == "torch") {
        modelPath /= "enet_b0_8_best_vgaf.pt";
    } else {
        modelPath /= "enet_b0_8_best_vgaf.onnx";
    }
    auto fer = EmotiEffLib::EmotiEffLibRecognizer::createInstance(backend, modelPath);
    fer->imgSize_ = 5;

    unsigned char data[3][5][5] = {{{82, 10, 160, 149, 128},
                                    {93, 34, 32, 169, 62},
                                    {114, 170, 48, 209, 104},
                                    {43, 197, 122, 157, 245},
                                    {165, 249, 118, 240, 116}},
                                   {{38, 53, 146, 243, 160},
                                    {55, 137, 67, 25, 197},
                                    {58, 70, 224, 235, 192},
                                    {53, 13, 83, 38, 32},
                                    {109, 151, 20, 50, 64}},
                                   {{71, 112, 222, 68, 226},
                                    {215, 198, 104, 73, 171},
                                    {144, 248, 253, 232, 66},
                                    {171, 197, 33, 15, 116},
                                    {161, 139, 3, 90, 126}}};

    int height = 5;
    int width = 5;
    // Create a cv::Mat in HWC format
    cv::Mat img(height, width, CV_8UC3);

    // Populate the cv::Mat with data
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            img.at<cv::Vec3b>(h, w) = cv::Vec3b(data[0][h][w], data[1][h][w], data[2][h][w]);
        }
    }

    auto res = fer->preprocess(img);

    xt::xarray<float> expTensor = {{{-0.7137, -1.9467, 0.6221, 0.4337, 0.0741},
                                    {-0.5253, -1.5357, -1.5699, 0.7762, -1.0562},
                                    {-0.1657, 0.7933, -1.2959, 1.4612, -0.3369},
                                    {-1.3815, 1.2557, -0.0287, 0.5707, 2.0777},
                                    {0.7077, 2.1462, -0.0972, 1.9920, -0.1314}},
                                   {{-1.3704, -1.1078, 0.5203, 2.2185, 0.7654},
                                    {-1.0728, 0.3627, -0.8627, -1.5980, 1.4132},
                                    {-1.0203, -0.8102, 1.8859, 2.0784, 1.3256},
                                    {-1.1078, -1.8081, -0.5826, -1.3704, -1.4755},
                                    {-0.1275, 0.6078, -1.6856, -1.1604, -0.9153}},
                                   {{-0.5670, 0.1476, 2.0648, -0.6193, 2.1346},
                                    {1.9428, 1.6465, 0.0082, -0.5321, 1.1759},
                                    {0.7054, 2.5180, 2.6051, 2.2391, -0.6541},
                                    {1.1759, 1.6291, -1.2293, -1.5430, 0.2173},
                                    {1.0017, 0.6182, -1.7522, -0.2358, 0.3916}}};

    expTensor = xt::expand_dims(expTensor, 0);

    EXPECT_TRUE(xt::allclose(res, expTensor, 1e-4, 1e-4));
}

std::string
TestNameGenerator(const ::testing::TestParamInfo<EmotiEffLibInternalTests::ParamType>& info) {
    auto backend = info.param;
    std::ostringstream name;
    name << "backend_" << backend;

    // Replace invalid characters for test names
    std::string name_str = name.str();
    std::replace(name_str.begin(), name_str.end(), '.', '_'); // Replace dots
    return name_str;
}

INSTANTIATE_TEST_SUITE_P(InternalMethods, EmotiEffLibInternalTests,
                         ::testing::ValuesIn(EmotiEffLib::getAvailableBackends()),
                         TestNameGenerator);
