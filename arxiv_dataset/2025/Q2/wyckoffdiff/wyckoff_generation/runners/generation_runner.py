import os
import time

import torch

from wyckoff_generation.common import utils
from wyckoff_generation.common.registry import registry
from wyckoff_generation.runners.base_runner import BaseRunner


@registry.register_runner("generate")
class GenerationRunner(BaseRunner):
    def __init__(self, config):
        super().__init__(config)
        self.num_samples_to_generate = config["num_samples"]
        self.save_dir = os.path.split(config["load"])[0]
        self.save_filename = f"generated_samples_N={self.num_samples_to_generate}_seed={config['seed']}.pt"

    def run(self):
        start_time = time.time()
        samples = self.model.generate(self.num_samples_to_generate)
        end_time = time.time()
        runtime = (end_time - start_time) / 3600
        print(
            f"Sampling {self.num_samples_to_generate} took {runtime} hours, or {end_time - start_time} seconds"
        )
        utils.save_generated_samples(samples, self.save_dir, self.save_filename)

    def init_model(self, config):
        checkpoint = utils.get_pretrained_checkpoint(config["load"])
        old_config = checkpoint["config"]
        old_config["device"] = self.device
        if old_config["mode"] == "train_d3pm":
            model = registry.get_model_class("d3pm")(old_config)
        else:
            ValueError("Don't know which class of generative model to load")

        model.load_state_dict(checkpoint["model_state_dict"])
        self.model = model.to(self.device)
        self.model.train(False)

    def init_dataloaders(self, config):
        # dont need this for now
        pass

    def init_optimizer(self, config):
        # dont need
        pass
