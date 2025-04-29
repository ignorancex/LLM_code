#include <trtengine.h>
#include "absl/flags/parse.h"
#include "absl/flags/flag.h"

ABSL_FLAG(std::string, onnx_path, "", "Path to onnx model file.");
ABSL_FLAG(std::string, engine_save_dir, "../models", "Path to the directory where the converted trt engines are stored.");
ABSL_FLAG(uint16_t, min_batch, 1, "Minimum batch size.");
ABSL_FLAG(uint16_t, max_batch, 128, "Max batch size");
ABSL_FLAG(uint16_t, step, 3, "Step to increase the batch size");
ABSL_FLAG(uint16_t, precision, 4, "Precision level FP32/FP16/INT8");
ABSL_FLAG(uint16_t, gpu, 0, "GPU Index");
ABSL_FLAG(size_t, max_workspace_size, 1 << 30, "Max workspace size for TRT layers.");

int main(int argc, char *argv[]) {
    absl::ParseCommandLine(argc, argv);

    std::string onnx_path = absl::GetFlag(FLAGS_onnx_path);
    std::string engine_save_dir = absl::GetFlag(FLAGS_engine_save_dir);
    uint16_t min_batch = absl::GetFlag(FLAGS_min_batch);
    uint16_t max_batch = absl::GetFlag(FLAGS_max_batch);
    uint16_t step = absl::GetFlag(FLAGS_step);
    uint16_t precision = absl::GetFlag(FLAGS_precision);
    int8_t gpu = absl::GetFlag(FLAGS_gpu);
    size_t max_workspace_size = absl::GetFlag(FLAGS_max_workspace_size);

    MODEL_DATA_TYPE prec = static_cast<MODEL_DATA_TYPE>(precision);

    uint16_t numConsecutiveFails = 0;
    bool success = true;

    for (uint16_t batch_size = min_batch; batch_size <= max_batch; batch_size += step) {
        std::cout << "====================================Converting batch size of " << batch_size << "====================================" << std::endl;
        TRTConfigs engineConfigs = {
            onnx_path,
            engine_save_dir,
            prec,
            "",
            128,
            1,
            batch_size,
            gpu,
            max_workspace_size
        };

        Engine engine(engineConfigs);
        success = engine.build();
        if (!success) {
            numConsecutiveFails += 1;
        } else {
            numConsecutiveFails = 0;
        }
        if (numConsecutiveFails >= 10) {
            throw std::runtime_error("To many consecutive conversion attempts have failed. Stop");
            return 1;
        }
        std::cout << "==================================================================================================" << std::endl;
    }
    return 0;
}