//
// Created by meiyixuan on 2024/4/15.
//
#include "../src/poller.h"

int main(int argc, char *argv[]) {
    // input
    // format: tcp://10.128.0.47:5555
    if (argc < 3) {
        std::cerr << "Too few parameters! [example: packed_client tcp://10.128.0.47:5555 tcp://10.128.0.48:5555]\n";
        return 1;
    }
    std::string server1 = argv[1];
    std::string server2 = argv[2];

    // ports and context
    std::vector<std::string> server_addresses = {
            server1,
            server2,
    };

    // initialize polling client
    zmq::context_t context(1);
    PollingClient client = PollingClient(context, server_addresses);
    while (true) {
        // poll to get message
        zmq::message_t buffer_msg;
        Header header = client.poll_once(buffer_msg, 10);
        auto receive_time = get_time();

        // check and print
        if (header.msg_type != MsgType::Invalid) {
            std::cout << "Received: " << std::endl;
            std::cout << "Creation time: " << header.creation_time << std::endl;
            std::cout << "Latency: " << receive_time - header.creation_time << " us\n";
            std::cout << "From server: " << header.request_id << std::endl;
            std::cout << std::endl;
        }
    }
}
