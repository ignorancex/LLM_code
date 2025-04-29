#include "sink_agent.h"

int main(int argc, char **argv) {
    absl::ParseCommandLine(argc, argv);
    auto *agent = new SinkAgent(absl::GetFlag(FLAGS_controller_url));

    std::string input;
    //wait for q as input to quit
    while (true) {
        std::cin >> input;
        if (input == "q") {
            break;
        }
    }

    delete agent;
    return 0;
}