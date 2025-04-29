#include "yolov8.h"

#include <utility>

YoloV8Agent::YoloV8Agent(const std::string &name, uint16_t own_port, int8_t devIndex, std::vector<Microservice*> services)
        : ContainerAgent(name, own_port, devIndex) {
    msvcs = std::move(services);
    std::thread preprocessor(&BaseReqBatcher::batchRequests, dynamic_cast<BaseReqBatcher*>(msvcs[1]));
    preprocessor.detach();
    std::thread inference(&BaseBatchInferencer::inference, dynamic_cast<BaseBatchInferencer*>(msvcs[2]));
    inference.detach();
    std::thread postprocessor(&BaseBBoxCropper::cropping, dynamic_cast<BaseBBoxCropper*>(msvcs[3]));
    postprocessor.detach();
    for (int i = 4; i < msvcs.size(); i++) {
        std::thread sender(&Sender::Process, dynamic_cast<Sender*>(msvcs[i]));
        sender.detach();
    }
}

int main(int argc, char **argv) {
    spdlog::set_pattern("[%C-%m-%d %H:%M:%S.%f] [%l] %v");

    absl::ParseCommandLine(argc, argv);

    int8_t device = absl::GetFlag(FLAGS_device);
    checkCudaErrorCode(cudaSetDevice(device), __func__);
    std::vector<BaseMicroserviceConfigs> msvc_configs = msvcconfigs::LoadFromJson();
    for (uint8_t i = 0; i < msvc_configs.size(); i++) {
        msvc_configs[i].msvc_deviceIndex = device;
    }

    std::string name = absl::GetFlag(FLAGS_name);
    uint16_t logLevel = absl::GetFlag(FLAGS_verbose);
    spdlog::set_level(spdlog::level::level_enum(logLevel));
    std::vector<Microservice*> msvcs;
    msvcs.push_back(new Receiver(msvc_configs[0]));
    msvcs.push_back(new BaseReqBatcher(msvc_configs[1]));
    msvcs[1]->SetInQueue(msvcs[0]->GetOutQueue());
    msvcs.push_back(new BaseBatchInferencer(msvc_configs[2]));
    msvcs[2]->SetInQueue(msvcs[1]->GetOutQueue());
    msvcs.push_back(new BaseBBoxCropper(msvc_configs[3]));
    msvcs[3]->SetInQueue(msvcs[2]->GetOutQueue());
    for (int i = 4; i < msvc_configs.size(); i++) {
        if (msvc_configs[i].msvc_dnstreamMicroservices.front().commMethod == CommMethod::localGPU) {
            msvcs.push_back(new GPUSender(msvc_configs[i]));
        } else if (msvc_configs[i].msvc_dnstreamMicroservices.front().commMethod == CommMethod::sharedMemory) {
            msvcs.push_back(new LocalCPUSender(msvc_configs[i]));
        } else if (msvc_configs[i].msvc_dnstreamMicroservices.front().commMethod == CommMethod::serialized) {
            msvcs.push_back(new RemoteCPUSender(msvc_configs[i]));
        }
        msvcs[i]->SetInQueue(msvcs[i - 1]->GetOutQueue());
    }
    ContainerAgent *agent = new YoloV8Agent(name, absl::GetFlag(FLAGS_port), device, msvcs);

    agent->waitReady();
    
    while (agent->running()) {
        std::this_thread::sleep_for(std::chrono::seconds(4));
        agent->SendQueueLengths();
    }
    delete agent;
    return 0;
}