import torch


class Aggregator:
    def __init__(self, args):
        self.verbose = args.verbose
        self.device = args.device
        self.server_todo = None

        self.running_times = []