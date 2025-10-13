import os

import pandas as pd
import torch

import wyckoff_generation.datasets.data_utils as data_utils
from wyckoff_generation.common.registry import registry
from wyckoff_generation.runners.base_runner import BaseRunner


@registry.register_runner("post_process")
class GenerationRunner(BaseRunner):
    def __init__(self, config):
        super().__init__(config)
        self.load_path = config["load"]
        self.post_process_all = config["post_process_all"]
        self.save_protostructures = config["save_protostructures"]
        self.post_processed_dir = "post_processed"
        self.checkpoints_path = "checkpoints"
        self.data_to_process = {}
        if self.post_process_all:
            print(f"Processing all unprocessed generated samples...")
            # Fetch all directories with data to process
            all_checkpoints = os.listdir(self.checkpoints_path)

            for checkpoint in all_checkpoints:
                checkpoint = os.path.join(self.checkpoints_path, checkpoint)
                data_files = self.fetch_data_files(checkpoint)
                if data_files is not None and data_files != []:
                    self.data_to_process[checkpoint] = data_files

        elif self.load_path is not None:
            data_files = []
            if os.path.isdir(self.load_path):
                data_files = self.fetch_data_files(self.load_path)
                if data_files is not None and data_files != []:
                    self.data_to_process[self.load_path] = data_files
            else:
                raise AssertionError(
                    f"Provided path to file '{self.load_path}'. Arguments provided with 'load' should be path to directory containing data files of format '.pt', e.g., a checkpoint directory."
                )

    def run(self):

        for data_dir, data_files in self.data_to_process.items():
            # Path to store updated dataset
            save_path = os.path.join(data_dir, self.post_processed_dir)

            for data_file_name in data_files:

                # Can run the updates on the post-processed dataset.
                if self.config["use_processed_data"]:
                    file_path = os.path.join(save_path, data_file_name)
                else:
                    file_path = os.path.join(data_dir, data_file_name)

                dataset = data_utils.load_dataset(file_path)

                print(f"Post processing '{file_path}'...")
                enriched_dataset = None
                if self.config["enrich_data"]:
                    # Enrich data set with protostructure, etc.
                    (
                        enriched_dataset,
                        protostructure_list,
                        prototype_list,
                    ) = data_utils.enrich_dataset(
                        dataset, return_protostructure_list=True
                    )

                # Save updates to processed dir. Note this must be done prior calling the novelty comparison.
                if enriched_dataset is not None:
                    self.save_processed(enriched_dataset, data_file_name, save_path)

                    if self.save_protostructures:
                        protostructure_file_path = os.path.join(
                            save_path,
                            f"{os.path.splitext(data_file_name)[0]}_protostructures.csv",
                        )
                        pd.DataFrame(
                            {
                                "protostructure": protostructure_list,
                                "prototype": prototype_list,
                            }
                        ).to_csv(protostructure_file_path, index=False)
                        print(
                            f"Exporting protostructures and prototypes, saved in '{protostructure_file_path}'."
                        )

    def save_processed(self, enriched_dataset, data_file_name, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        torch.save(enriched_dataset, os.path.join(save_path, data_file_name))

    def init_model(self, config):
        # Don't need
        pass

    def fetch_data_files(self, dir_path):
        data_files = []
        checkpoint_contents = os.listdir(dir_path)
        if checkpoint_contents == []:
            print(f"Checkpoint '{dir_path}' is empty directory. Skipping... ")
            return None
        if self.post_processed_dir in checkpoint_contents:
            if self.config["use_processed_data"]:
                # Use the processed dir to update datafile
                dir_path = os.path.join(dir_path, self.post_processed_dir)
            else:
                print(
                    f"Checkpoint '{dir_path}' already has post processed data in '{self.post_processed_dir}'. Use flag '--use_processed_data', if wanting to run on already processed files. Skipping... "
                )
                return None
        for item in checkpoint_contents:
            item_path = os.path.join(dir_path, item)
            if os.path.isfile(item_path):
                root, ext = os.path.splitext(item_path)
                if ext == ".pt" and "generated_samples" in item:
                    print(
                        f"Adding checkpoint '{dir_path}' data file '{item}' to post processing list."
                    )
                    data_files.append(item)
                else:
                    print(
                        f"Found file '{item}' in directory '{dir_path}', but was not pytorch dataformat .pt. Skipping..."
                    )
        return data_files

    def init_dataloaders(self, config):
        # dont need this for now
        pass

    def init_optimizer(self, config):
        # dont need
        pass
