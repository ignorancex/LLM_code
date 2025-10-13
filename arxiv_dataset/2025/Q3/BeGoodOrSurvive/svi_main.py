import traceback
import time
import os
import sys
sys.path.insert(0, '/scratch/10384/dylantw15/Bayesian-Neat-Project')
import datetime
# Print the current working directory and sys.path
#print("Current Working Directory:", os.getcwd())
#print("Python Path:", sys.path)

# Add the current directory to sys.path explicitly
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
import json
import torch
import re

import bnn.bayesnn
from bnn.bayesnn import BayesianNN

import numpy as np
import matplotlib.pyplot as plt
import pyro
import torch.multiprocessing as mp

from svi_loops import main_loop, generational_driver
from utils.plotting import plot_loss_and_survival, plot_survival_and_ethics, plot_loss_and_ethics
from utils.utils_logging import save_experiment_results
from neat.neat_evolution import NeatEvolution
from utils.text_generation import generate_text
from utils.text_utils import normalize_string, trim_response, extract_choices_and_intro
from bnn.bnn_utils import update_bnn_history

import sys

import bnn_neat
from bnn_neat.genome import DefaultGenome

model = "gpt-4o"

from mpi4py import MPI
# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


if __name__ == "__main__":
    start_time = time.time()
    try:
        # Initialize configurations (if needed by all ranks)
        config_path = "config-feedforward"
        config = bnn_neat.config.Config(
            bnn_neat.DefaultGenome,
            bnn_neat.DefaultReproduction,
            bnn_neat.DefaultSpeciesSet,
            bnn_neat.DefaultStagnation,
            config_path
        )
        # Initialize the genome and BNN on all ranks
        genome_id = 0  # Or use a different ID for each rank if needed
        genome = DefaultGenome(genome_id)
        genome.configure_new(config.genome_config)

        # Initialize the BNN with the genome and config
        bnn = BayesianNN(genome, config)

        # Initialize NeatEvolution on all ranks
        neat_trainer = None

        print("Rank = 0 in main", flush=True)
        # Only Rank 0 proceeds with the main logic
        max_tokens = 10240
        temperature = 1.2
        top_p = 0.95
        danger = 2

        pyro.enable_validation(False)

        # Proceed with Rank 0's main logic
        votes = {'strong': 0, 'weak': 10}
        shared_history = []
        bnn_history = []
        ground_truth_label_list = []
        ethical_ground_truths = []
        gen_loss_history = []
        gen_ethical_history = []
        num_gens = 90
        global_counter = 0
        # Call the loop logic
        result, gen_loss_history, gen_ethical_history, ethical_ground_truths, survival_ground_truths = generational_driver(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            danger=danger,
            shared_history=shared_history,
            bnn_history=bnn_history,
            ground_truth_label_list=ground_truth_label_list,
            ethical_ground_truths=ethical_ground_truths,
            strong_bnn=bnn,
            config=config,
            num_gens=num_gens,
            neat_trainer=neat_trainer,
            global_counter=global_counter,
            comm=comm
        )

        print("Experiment complete. Results saved.")

    except Exception as e:
        # Print or log the exception with traceback
        error_message = f"Exception in main.py: {e}"
        traceback_details = traceback.format_exc()  # Get the full traceback as a string
        print(error_message)
        print(traceback_details)

    finally:
        # Stop the timer and calculate elapsed time
        elapsed_time = time.time() - start_time
        print(f"Total runtime: {elapsed_time:.2f} seconds")

        # Flush standard output and error
        sys.stdout.flush()
        sys.stderr.flush()



