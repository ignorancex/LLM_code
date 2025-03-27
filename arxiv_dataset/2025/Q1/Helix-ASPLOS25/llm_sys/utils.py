# 2024.04.24 Yixuan Mei
import torch
import socket

# in simulator, the first compute node has idx 2
# in real sys, the first compute node has idx 1
# simulator idx - SIMULATOR_NODE_OFFSET = real node idx
SIMULATOR_NODE_OFFSET = 1


def to_real(node_id: int) -> int:
    if node_id == 0 or node_id == 1:
        return 0
    else:
        return node_id - SIMULATOR_NODE_OFFSET


CONFIG_BROADCAST_ADDR = "tcp://10.128.0.31:5000"


def warm_up():
    # create a tensor and move it to GPU (Warm up GPU)
    x = torch.tensor([1, 2, 3])
    for i in range(100):
        x.cuda()


def get_local_ip():
    # Attempt to connect to an internet host in order to determine the local interface
    try:
        # Create a dummy socket to connect to an Internet IP or DNS
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            # Use Google's public DNS server to find out our IP
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        return ip
    except Exception as e:
        return f"Error obtaining local IP: {str(e)}"


class FlyingQuery:
    def __init__(self, query_uid, input_length, output_length, compute_node_uids, start_layers, end_layers, pipeline):
        self.query_uid = query_uid
        self.input_length = input_length
        self.output_length = output_length
        self.processed_tokens = 0

        # scheduling
        self.compute_node_uids = compute_node_uids
        self.start_layers = start_layers
        self.end_layers = end_layers
        # List[PipelineStage]
        self.pipeline = pipeline
