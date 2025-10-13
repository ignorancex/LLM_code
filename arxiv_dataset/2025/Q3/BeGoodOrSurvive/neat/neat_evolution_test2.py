import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
import random
import logging
import subprocess
import torch
import sys
import traceback
import subprocess
import datetime

def log_gpu_utilization(rank=None):
    gpu_id = os.getenv("CUDA_VISIBLE_DEVICES", "0")
    result = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                             "--format=csv,noheader,nounits", "-i", gpu_id],
                            stdout=subprocess.PIPE, text=True)
    utilization, memory = result.stdout.strip().split(", ")
    rank_info = f"Rank {rank}: " if rank is not None else ""
    print(f"GPU {gpu_id}: Utilization={utilization}%, Memory Used={memory} MB")

def handle_exception(exc_type, exc_value, exc_traceback):
    if not issubclass(exc_type, KeyboardInterrupt):
        print("Unhandled Exception", flush=True)
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr)

sys.excepthook = handle_exception

import numpy as np
from math import ceil

import pickle
import numpy as np
import logging


import torch.nn as nn

import time
import random
import sys
from neat.neat_utils import save_evolution_results
from bnn.bayesnntest2 import BayesianNN
import json
import pyro
import pyro.infer

import bnn_neat
from bnn_neat.checkpoint import Checkpointer
from bnn_neat.config import Config
from bnn_neat.genome import DefaultGenome
from bnn_neat.reproduction import DefaultReproduction
from bnn_neat.species import DefaultSpeciesSet
from bnn_neat.stagnation import DefaultStagnation
from bnn_neat.genes import DefaultConnectionGene, DefaultNodeGene
from bnn_neat.reporting import ReporterSet, StdOutReporter, BaseReporter



class BayesianGenome(DefaultGenome):
    # Extend DefaultGenome to handle weight_mu and weight_sigma
    def __init__(self, key):
        super().__init__(key)
        # Additional initialization if needed

    # Override methods if necessary to handle weight_mu and weight_sigma



class NeatEvolution:
    def __init__(self, config, config_path, bnn, neat_iteration=None, comm=None):

        if config is None:
            raise ValueError("Config cannot be None.")
        if not isinstance(config_path, str) or not config_path:
            raise ValueError("Config path must be a valid string.")
        if bnn is None:
            raise ValueError("BNN instance cannot be None.")

        # MPI initialization
        from mpi4py import MPI
        self.comm = comm or MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Local communicator for splitting nodes
        if self.size == 1:
            # Single rank fallback
            self.local_comm = self.comm
            self.local_rank = 0
        else:
            try:
                self.local_comm = self.comm.Split_type(MPI.COMM_TYPE_SHARED)
                self.local_rank = self.local_comm.Get_rank()
            except Exception as e:
                print(f"Rank {self.rank}: Error in Split_type - {e}", flush=True)
                self.local_comm = None
                self.local_rank = -1

        self.gpu_count = torch.cuda.device_count()


        self.gpu_assignments = self.assign_gpus_to_ranks()


        # Synchronize all ranks before proceeding
        comm.Barrier()

        import logging

        try:
            # Setup logging
            logging.basicConfig(
                filename=f"log_rank_{self.rank}.txt",
                level=logging.DEBUG,
                format="%(asctime)s - %(levelname)s - %(message)s",
                filemode="w"
            )
            handler = logging.FileHandler(f"prod_log_rank_{self.rank}.txt", mode="w")
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)

            # Create a logger specific to this rank
            logger = logging.getLogger(f"Rank_{self.rank}")
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)  # Explicitly set the logger's level

        except Exception as e:
            print(f"Rank {self.rank}: Error setting up logging: {e}", flush=True)

        # Shared configuration
        self.config = config
        self.config_path = config_path
        self.bnn = bnn
        self.stagnation_limit = 235
        self.neat_iteration = neat_iteration
        self.population_tradeoffs = []
        self.best_fitness = None
        self.generations_without_improvement = 0

        # Extract BNN parameters
        num_inputs, num_outputs, connections, attention_layers = self.extract_bnn_parameters(bnn)
        self.attention_layers = attention_layers

        # Synchronize attention layer state for all ranks
        self.comm.Barrier()

        # Update NEAT configuration for all ranks
        self.update_neat_config(config, num_inputs, num_outputs)

        # Only Rank 0 performs population initialization
        if self.rank == 0:
            print(f"Rank {self.rank}: Initializing population...", flush=True)
            optimized_params = bnn.get_optimized_parameters()
            initial_population = self.create_initial_population(connections, optimized_params)
            self.population = bnn_neat.Population(config)
            self.population.population = initial_population
            self.population.species.speciate(config, self.population.population, self.population.generation)
            self.population.add_reporter(bnn_neat.StdOutReporter(True))
            self.stats = bnn_neat.StatisticsReporter()
            self.population.add_reporter(self.stats)
            print(f"Rank {self.rank}: Population initialized", flush=True)
            self.checkpointer = Checkpointer(generation_interval=5, time_interval_seconds=600, filename_prefix="neat_checkpoints/neat-checkpoint-")
            self.population.add_reporter(self.checkpointer)
        else:
            self.population = None

        # Synchronize all ranks after initialization
        self.comm.Barrier()

    def restore_from_checkpoint(self, checkpoint_path):
        if self.rank == 0:
            self.population = Checkpointer.restore_checkpoint(checkpoint_path)
            print(f"Rank {self.rank}: Restored population from checkpoint: {checkpoint_path}", flush=True)
        else:
            self.population = None

        # Synchronize all ranks after restoration
        self.comm.Barrier()

    def set_device_for_current_rank(self):
        """
        Set the CUDA_VISIBLE_DEVICES environment variable and torch device for the current rank.
        """
        assigned_gpu = self.gpu_assignments[self.rank]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(assigned_gpu.split(":")[1])  # Extract GPU index
        torch.cuda.set_device(0)  # Map to the first visible GPU

    def assign_gpus_to_ranks(self):
        """
        Create a mapping of ranks to GPU devices.
        """
        self.gpu_assignments = {}
        for rank in range(self.size):
            assigned_gpu = self.assign_gpu_round_robin(rank, self.gpu_count)
            self.gpu_assignments[rank] = f"cuda:{assigned_gpu}"
        return self.gpu_assignments

    def assign_gpu_round_robin(self, rank, num_gpus):
        """
        Assign GPUs to ranks using a round-robin strategy.
        """
        assigned_gpu = rank % num_gpus
        return assigned_gpu

    def update_neat_config(self, config, num_inputs, num_outputs):
        """
        Updates the NEAT configuration to match the BNN's input and output sizes.
        """
        config.genome_config.num_inputs = num_inputs
        config.genome_config.num_outputs = num_outputs
        #config.genome_config.initial_connection = 'full'  # Adjust as needed


    def extract_bnn_parameters(self, bnn):
        """
        Extracts the necessary parameters from the BayesianNN object.

        Returns:
            num_inputs (int): Number of input nodes.
            num_outputs (int): Number of output nodes.
            hidden_layers (list): List containing the number of nodes in each hidden layer.
            connections (dict): Dictionary of connections with their properties.
        """
        # Extract the number of input and output nodes
        num_inputs = bnn.config.genome_config.num_inputs
        num_outputs = bnn.config.genome_config.num_outputs
        # Extract hidden layer configurations
        # Assuming bnn.hidden_layers is a list like [64, 32] for two hidden layers
        #hidden_layers = bnn.config.genome_config.hidden_layers MIGHT have to come back to this? I cant tell
        hidden_layers = []

        # Extract connections
        connections = bnn.get_connections()

        # Extract attention layers (query, key, value projections)
        attention_layers = {
            'query_proj': bnn.query_proj.state_dict(),
            'key_proj': bnn.key_proj.state_dict(),
            'value_proj': bnn.value_proj.state_dict()
        }

        return num_inputs, num_outputs, connections, attention_layers

    def create_initial_population(self, connections, optimized_params):
        """
        Creates the initial NEAT population based on the BNN's connections.

        Args:
            connections (dict): Connections from the BNN.

        Returns:
            population (dict): A dictionary of genomes for NEAT's initial population.
        """
        population = {}
        for i in range(self.config.pop_size):
            genome_id = i
            genome = self.create_genome_from_bnn(genome_id, connections, optimized_params)
            population[genome_id] = genome
        return population

    def create_genome_from_bnn(self, genome_id, connections, optimized_params):
        """
        Creates a NEAT genome initialized with BNN's connections and node parameters.
        
        Args:
            genome_id (int): The ID of the genome.
            connections (dict): Dictionary of connection data from the BNN.
                Expected keys: (in_node, out_node) tuples
                Each value should be a dict with keys like:
                   'weight_mu', 'weight_sigma', 'enabled'
            optimized_params (dict): A dictionary of optimized parameters (weights, biases).
                Keys are strings like 'w_mu_(in,out)', 'w_sigma_(in,out)', 'b_mu_nodeId', 'b_sigma_nodeId',
                'r_mu_nodeId', 'r_sigma_nodeId' for response parameters, etc.
                
        Returns:
            genome: A NEAT genome object of type BayesianGenome (or similar) with fully initialized genes.
        """
        genome = BayesianGenome(genome_id)
        genome.configure_new(self.config.genome_config)

        # Initialize connection genes
        for conn_key, conn_data in connections.items():
            conn_gene = DefaultConnectionGene(conn_key)

            # Handle weight_mu
            weight_mu_name = f"w_mu_{conn_key}"
            if weight_mu_name in optimized_params:
                conn_gene.weight_mu = optimized_params[weight_mu_name].cpu().squeeze().item()
            else:
                conn_gene.weight_mu = conn_data.get('weight_mu', 0.0)  # Provide a default if needed

            # Handle weight_sigma
            weight_sigma_name = f"w_sigma_{conn_key}"
            if weight_sigma_name in optimized_params:
                conn_gene.weight_sigma = optimized_params[weight_sigma_name].cpu().squeeze().item()
            else:
                conn_gene.weight_sigma = conn_data.get('weight_sigma', 1.0)  # Provide a default if needed

            # Handle enabled
            conn_gene.enabled = conn_data.get('enabled', True)

            genome.connections[conn_key] = conn_gene

        # Initialize node genes from the BNN's nodes
        # We assume self.bnn.nodes is a dict keyed by node_id with values:
        # { 'bias_mu': ..., 'bias_sigma': ..., 'response_mu': ..., 'response_sigma': ..., 
        #   'activation': 'relu', 'aggregation': 'sum', ... }
        # If these keys are not present, please add defaults or handle accordingly.

        for node_id, node_data in self.bnn.nodes.items():
            # Skip input nodes if needed, depends on your logic
            # If input nodes need parameters, remove this check
            if node_id < 0:
                continue

            node_gene = genome.nodes.get(node_id)
            if node_gene is None:
                node_gene = DefaultNodeGene(node_id)
                genome.nodes[node_id] = node_gene

            # Handle bias_mu
            bias_mu_name = f"b_mu_{node_id}"
            if bias_mu_name in optimized_params:
                node_gene.bias_mu = optimized_params[bias_mu_name].cpu().squeeze().item()
            else:
                node_gene.bias_mu = node_data.get('bias_mu', 0.0)  # Default if needed

            # Handle bias_sigma
            bias_sigma_name = f"b_sigma_{node_id}"
            if bias_sigma_name in optimized_params:
                node_gene.bias_sigma = optimized_params[bias_sigma_name].cpu().squeeze().item()
            else:
                node_gene.bias_sigma = node_data.get('bias_sigma', 1.0)  # Default if needed

            # Handle response_mu
            response_mu_name = f"r_mu_{node_id}"
            if response_mu_name in optimized_params:
                node_gene.response_mu = optimized_params[response_mu_name].cpu().squeeze().item()
            else:
                node_gene.response_mu = node_data.get('response_mu', 0.0)  # Default if needed

            # Handle response_sigma
            response_sigma_name = f"r_sigma_{node_id}"
            if response_sigma_name in optimized_params:
                node_gene.response_sigma = optimized_params[response_sigma_name].cpu().squeeze().item()
            else:
                node_gene.response_sigma = node_data.get('response_sigma', 1.0)  # Default if needed

            # Handle activation
            # If node_data does not provide activation, pick a default from options: 'relu', 'sigmoid', 'tanh'
            node_gene.activation = node_data.get('activation', 'relu')

            # Handle aggregation
            # If node_data does not provide aggregation, pick a default from options: 'sum', 'product', 'max', 'min', 'mean'
            node_gene.aggregation = node_data.get('aggregation', 'sum')

        return genome

    def run_neat_step(self, strong_bnn, bnn_history, ground_truth_labels, ethical_ground_truths, comm):
        print(f"Rank {self.rank}: Starting NEAT run step", flush=True)
        self.max_generations = 10
        # Broadcast the population to all ranks
        if self.rank == 0:
            # Serialize the population on rank 0
            serialized_population = pickle.dumps(self.population)
        else:
            serialized_population = None

        # Broadcast serialized population to all ranks
        serialized_population = self.comm.bcast(serialized_population, root=0)

        import traceback

        # Deserialize the population on all ranks
        self.population = pickle.loads(serialized_population)
        try:
            winner = self.population.run(
                lambda genomes, config, k: self.fitness_function(
                    genomes, config, k,
                    bnn_history=bnn_history,
                    ground_truth_labels=ground_truth_labels,
                    ethical_ground_truths=ethical_ground_truths,
                    k=k, 
                    bnn=strong_bnn,
                ),
                n=self.max_generations,
                neat_iteration=self.neat_iteration, 
                comm=comm,
                max_gens = self.max_generations
            )
        except Exception as e:
            print(f"Rank {self.rank}: Error in run_neat_step: {e}", flush=True)
            traceback.print_exc()  # Print the full traceback
            self.comm.Abort()
            raise

        if self.rank == 0:
            print("Rank 0: NEAT evolution completed.", flush=True)
            return winner
        else:
            return None  # Workers do not return the winner

    def fitness_function(self, genomes, config, neat_iteration, bnn_history, ground_truth_labels, ethical_ground_truths, k, bnn):
        """
        Evaluate the fitness of genomes using MPI for parallel processing.
        This function centralizes all MPI logic.
        """
        comm = self.comm
        rank = self.rank
        size = self.size

        # Initialize evolution results if not already defined
        if not hasattr(self, "evolution_results"):
            self.evolution_results = {
                "fitness_summary": [],  # Holds per-generation fitness statistics
                "population_tradeoffs": []  # Stores decision and ethical histories
            }

        try:
            # Rank 0 prepares data to send to workers
            if rank == 0:
                # Prepare attention layers' state_dicts
                attention_layers_state_dict = self.attention_layers

                # Prepare data to broadcast
                data_to_send = {
                    'bnn_history': bnn_history,
                    'ground_truth_labels': ground_truth_labels,
                    'ethical_ground_truths': ethical_ground_truths,
                    'attention_layers_state_dict': attention_layers_state_dict
                }
            else:
                data_to_send = None

            # Broadcast data to all ranks
            data_received = comm.bcast(data_to_send, root=0)

            # Unpack received data
            bnn_history = data_received['bnn_history']
            ground_truth_labels = data_received['ground_truth_labels']
            ethical_ground_truths = data_received['ethical_ground_truths']
            attention_layers_state_dict = data_received['attention_layers_state_dict']

            # Rank 0 distributes genomes to workers
            if rank == 0:
                genomes_list = list(genomes)
                total_genomes = len(genomes_list)
                genomes_per_worker = total_genomes // size
                remainder = total_genomes % size

                assigned_genomes = []
                start_idx = 0
                for worker_rank in range(size):
                    end_idx = start_idx + genomes_per_worker + (1 if worker_rank < remainder else 0)
                    assigned_genomes.append(genomes_list[start_idx:end_idx])
                    start_idx = end_idx


                    for w_rank, worker_genomes in enumerate(assigned_genomes):
                        assigned_device = self.gpu_assignments[w_rank]
                        for genome_id, genome in worker_genomes:
                            genome.device = assigned_device  # Assign device to genome
                            try:
                                if genome.device is None:
                                    raise ValueError(f"Genome {genome_id} does not have a valid device: {genome.device}")
                            except Exception as e:
                                raise ValueError(f"Failed to create device for genome {genome_id}: {genome.device}, Error: {e}")


                # Send genomes to worker ranks
                for worker_rank in range(1, size):
                    genomes_data = pickle.dumps(assigned_genomes[worker_rank])
                    comm.send(genomes_data, dest=worker_rank, tag=11)

                 # Prepare local genomes for Rank 0
                local_genomes = assigned_genomes[0]

            else:
                # Worker ranks receive genomes
                genomes_data = comm.recv(source=0, tag=11)
                local_genomes = pickle.loads(genomes_data)

            comm.Barrier()

            genome_ids = [genome_id for genome_id, _ in local_genomes]

            # Step 2: Evaluate genomes locally
            local_results = self.evaluate_genomes(
                local_genomes, config, self.attention_layers,
                bnn_history, ground_truth_labels, ethical_ground_truths
            )

            # Step 2: Apply `k > 5` logic locally
            if k > 5:
                # Sort local genomes based on fitness
                sorted_genomes = sorted(local_results, key=lambda x: x[1], reverse=True)
                top_percentage = 0.2  # Top 20%
                top_count = max(1, int(len(sorted_genomes) * top_percentage))
                top_genome_ids = {g[0] for g in sorted_genomes[:top_count]}  # Set of top genome IDs
            else:
                top_genome_ids = set()

            # Update evaluation_window for each genome
            for genome_id, fitness, genome in local_results:
                if k <= 5:  # Early exploration phase
                    genome.evaluation_window = 30
                elif 5 < k <= 15:  # Intermediate phase
                    if genome_id in top_genome_ids:
                        genome.evaluation_window = 30
                    else:
                        genome.evaluation_window = 30
                elif k > 15:  # Later phase with more refined genomes
                    if genome_id in top_genome_ids:
                        genome.evaluation_window = 30
                    else:
                        genome.evaluation_window = 30

            # Step 3: Prepare results to send back to Rank 0
            # Update local_results to include the updated genomes
            local_results_to_send = [(genome_id, genome.fitness, genome) for genome_id, fitness, genome in local_results]
            if rank == 0:
                # Rank 0 collects results from workers
                all_results = local_results
                for worker_rank in range(1, size):
                    worker_results_data = comm.recv(source=worker_rank, tag=22)
                    worker_results = pickle.loads(worker_results_data)
                    all_results.extend(worker_results)

                # Create the updated population
                updated_population = {genome_id: genome for genome_id, fitness, genome in all_results}

                # Iterate over the original genomes and update them
                for genome_id, genome in genomes:
                    if genome_id in updated_population:
                        updated_genome = updated_population[genome_id]

                        # Update the original genome's attributes
                        genome.fitness = updated_genome.fitness
                        genome.evaluation_window = updated_genome.evaluation_window
                        genome.decision_history = updated_genome.decision_history
                        genome.ethical_score_history = updated_genome.ethical_score_history
                        genome.mutation_history = updated_genome.mutation_history


            else:
                # Worker ranks send their local results
                results_data = pickle.dumps(local_results)
                comm.send(results_data, dest=0, tag=22)
                updated_population = None

            comm.Barrier()


            if rank == 0:
                fitness_report = []
                decision_histories = []
                for genome_id, genome in genomes:
                    fitness_report.append(genome.fitness)

                    # Collect the decision and ethical histories
                    decision_histories.append({
                        'genome_id': genome_id,
                        'decisions': list(genome.decision_history),
                        'ethical_scores': list(genome.ethical_score_history),
                        'fitness': float(genome.fitness),
                        'mutation_history': list(genome.mutation_history)
                    })

                # Store the collected data for this generation
                self.population_tradeoffs.append({
                    'generation': k,
                    'tradeoffs': decision_histories
                })

                # Calculate summary statistics
                mean_fitness = np.mean(fitness_report)
                median_fitness = np.median(fitness_report)
                std_fitness = np.std(fitness_report)
                upper_q = np.percentile(fitness_report, 75)
                lower_q = np.percentile(fitness_report, 25)
                iqr_fitness = upper_q - lower_q

                # Get the top 5 and bottom 5 fitness scores
                sorted_fitness = sorted(fitness_report, reverse=True)
                top_5_fitness = sorted_fitness[:5]
                bottom_5_fitness = sorted_fitness[-5:]

                # Find current best fitness
                current_best_fitness = max(fitness_report)

                # Summarize fitness
                fitness_summary = {
                    'generation': k,
                    'mean_fitness': mean_fitness,
                    'median_fitness': median_fitness,
                    'std_fitness': std_fitness,
                    'upper_quartile': upper_q,
                    'lower_quartile': lower_q,
                    'iqr_fitness': iqr_fitness,
                    'top_5_fitness': top_5_fitness,
                    'bottom_5_fitness': bottom_5_fitness,
                    'best_fitness': current_best_fitness
                }
                self.evolution_results["fitness_summary"].append(fitness_summary)

                if k == self.max_generations:
                    save_evolution_results(self.evolution_results, self.population_tradeoffs, neat_iteration = self.neat_iteration)

                # Check for fitness improvement
                if self.best_fitness is None or current_best_fitness > self.best_fitness:
                    self.best_fitness = current_best_fitness
                    self.generations_without_improvement = 0
                    print(f"New best fitness: {self.best_fitness:.4f}")
                else:
                    self.generations_without_improvement += 1
                    print(f"No improvement in fitness for {self.generations_without_improvement} generations")

                # Check for stagnation
                if self.generations_without_improvement >= self.stagnation_limit:
                    print(f"Stopping evolution: No improvement in fitness for {self.stagnation_limit} generations.")
                    save_evolution_results(self.evolution_results, self.population_tradeoffs, neat_iteration = self.neat_iteration)
                    should_stop = True
                else:
                    should_stop = False

            else:
                should_stop = None

            # Broadcast the decision to all processes
            should_stop = comm.bcast(should_stop, root=0)
            comm.Barrier()
            return should_stop

        except Exception as e:
            logging.error(f"Rank {rank}: Error during fitness function: {e}")
            traceback.print_exc()
            comm.Abort()
            raise

    def evaluate_genomes(self, genomes, config, attention_layers_state_dict,
                         bnn_history, ground_truth_labels, ethical_ground_truths):
        """
        Evaluate a batch of genomes and return their fitness results.
        """
        results = []

        for genome_id, genome in genomes:
            try:
                if genome.device is None:
                    raise ValueError(f"Genome {genome_id} does not have a valid device: {genome.device}")

                assigned_device = self.gpu_assignments[self.rank]  # Access pre-assigned GPU for the current rank
                device = torch.device(assigned_device)

                # Build model from genome
                local_bnn = BayesianNN(genome, config)
                local_bnn.build_network(config)

                # Load attention layers
                local_bnn.query_proj.load_state_dict(attention_layers_state_dict['query_proj'])
                local_bnn.key_proj.load_state_dict(attention_layers_state_dict['key_proj'])
                local_bnn.value_proj.load_state_dict(attention_layers_state_dict['value_proj'])

                bnn_history_device = self.move_bnn_history_to_device(bnn_history, device)

                # Evaluate genome
                fitness = self.evaluate_genome(genome_id, genome, local_bnn, config,
                                               bnn_history_device, ground_truth_labels, ethical_ground_truths, device)

                results.append((genome_id, fitness, genome))

                # Clean up
                del local_bnn
                torch.cuda.empty_cache()
            except Exception as e:
                traceback.print_exc()
                raise
                fitness = float('-inf')
                results.append((genome_id, fitness, genome))
        return results

    def evaluate_genome(self, genome_id, genome, bnn, config,
                    bnn_history, ground_truth_labels, ethical_ground_truths, device):
        """
        Evaluate a single genome and return its fitness.
        """
        try:
            bnn_history_device = self.move_bnn_history_to_device(bnn_history, device)

            # Reset the BNN's input matrix and last update index
            bnn.input_matrix = None
            bnn.last_update_index = 0

            # Initialize fitness components
            total_loss = 0.0
            num_entries = 0

            # Define evaluation window
            evaluation_window = getattr(genome, 'evaluation_window', 20)  # Default to 20 if not set

            # Sample entries for evaluation
            entries_to_evaluate = self.sample_entries(bnn_history_device, evaluation_window)

            # Merge list of dictionaries into a single dictionary
            ground_truth_labels_dict = {}
            for dictionary in ground_truth_labels:
                ground_truth_labels_dict.update(dictionary)

            # Merge list of dictionaries into a single dictionary
            ethical_ground_truths_dict = {}
            for dictionary in ethical_ground_truths:
                ethical_ground_truths_dict.update(dictionary)

            decision_history = []
            ethical_score_history = []

            for idx, entry in entries_to_evaluate:
                entry_id = entry.get('id')
                expected_output = ground_truth_labels_dict.get(entry_id)
                ethical_scores = ethical_ground_truths_dict.get(entry_id)

                if expected_output is None or ethical_scores is None:
                    continue  # Skip if data is missing

                expected_output_tensor = torch.tensor(expected_output, dtype=torch.float32, device=device).unsqueeze(0)

                # Compute loss and get predictions
                loss, logits, probabilities = self.compute_bce_loss(
                    bnn, bnn_history_device[:idx + 1], expected_output_tensor, current_index=idx, device=device
                )

                total_loss += loss.item()
                num_entries += 1

                # Normalize probabilities using softmax to ensure they sum to 1
                normalized_probabilities = torch.softmax(probabilities, dim=0)

                # Sample action based on the normalized probabilities
                sampled_choice = torch.multinomial(normalized_probabilities, num_samples=1).item()
                decision_history.append((idx, sampled_choice))

                # Get the ethical score corresponding to the sampled action
                ethical_score = ethical_scores[sampled_choice]
                ethical_score_history.append((idx, ethical_score))


            if num_entries > 0:
                average_loss = total_loss / num_entries
            else:
                average_loss = float('inf')  # High loss if no valid entries

            fitness = -average_loss  # Assuming lower loss is better

            # Assign fitness to genome
            genome.fitness = fitness
            # Attach the decision and ethical score histories to the genome
            genome.decision_history.extend(decision_history)
            genome.ethical_score_history.extend(ethical_score_history)


            return fitness

        except Exception as e:
            print(f"Error in evaluating genome {genome_id}: {e}")
            traceback.print_exc()
            raise

    def move_bnn_history_to_device(self, bnn_history, device):
        bnn_history_device = []
        for entry in bnn_history:
            entry_device = {}
            for key, value in entry.items():
                try:
                    if isinstance(value, torch.Tensor):
                        entry_device[key] = value.to(device)
                    elif isinstance(value, np.ndarray):
                        entry_device[key] = torch.tensor(value, device=device, dtype=torch.float)
                    elif isinstance(value, list) and all(isinstance(v, (int, float)) for v in value):
                        entry_device[key] = torch.tensor(value, device=device, dtype=torch.float)
                    else:
                        # Leave non-tensorizable data (e.g., strings) as-is
                        entry_device[key] = value
                except Exception as e:
                    print(f"Error processing key {key} with value {value}: {e}")
                    raise
            bnn_history_device.append(entry_device)
        return bnn_history_device

    def sample_entries(self, bnn_history, evaluation_window):
        """
        Samples an even number of entries from 'easy', 'medium', and 'hard' difficulty levels,
        while introducing randomness to avoid always sampling the same entries.
        """
        # Track selected samples and ensure balanced difficulty distribution
        selected_entries = []
        sampling_splits = {"easy": 0, "medium": 0, "hard": 0}
        max_per_difficulty = evaluation_window // 3  # Target per difficulty level

        # Difficulty ranges
        difficulty_map = {
            "easy": [1, 2, 3],
            "medium": [4, 5, 6],
            "hard": [7, 8, 9, 10]
        }

        # Group entries by difficulty
        difficulty_buckets = {"easy": [], "medium": [], "hard": []}
        for idx, entry in enumerate(bnn_history):
            if entry['agent'] != 'Storyteller':
                continue  # Skip non-storyteller entries

            difficulty = entry.get('difficulty')
            if difficulty is None:
                continue  # Skip if difficulty is missing

            for category, range_values in difficulty_map.items():
                if difficulty in range_values:
                    difficulty_buckets[category].append((idx, entry))
                    break

        # Randomly sample entries from each difficulty bucket
        for difficulty, entries in difficulty_buckets.items():
            num_to_sample = min(max_per_difficulty, len(entries))
            selected_entries.extend(random.sample(entries, num_to_sample))
            sampling_splits[difficulty] += num_to_sample

        # If we still don't have enough entries, randomly sample from all storyteller entries
        if len(selected_entries) < evaluation_window:
            remaining_slots = evaluation_window - len(selected_entries)
            storyteller_entries = [(idx, entry) for idx, entry in enumerate(bnn_history) if entry['agent'] == 'Storyteller']
            extra_samples = random.sample(
                storyteller_entries, min(remaining_slots, len(storyteller_entries))
            )
            selected_entries.extend(extra_samples)

        # Debugging info (remove in production)
        if self.rank == 0:
            print(f"Final Sampling Splits: {sampling_splits}")

        return selected_entries

    def compute_bce_loss(self,bnn, bnn_history, ground_truth_labels, current_index=None, device=None):
        """
        Compute Binary Cross-Entropy (BCE) loss for the model.

        Parameters:
        - bnn_history (list): History of BNN states.
        - ground_truth_labels (torch.Tensor): Ground truth labels for comparison.
        - current_index (int, optional): The index of the current data point in the history. Defaults to None.
        - device (torch.device, optional): The device to perform computations on. Defaults to GPU if available, otherwise CPU.

        Returns:
        - tuple: A tuple containing (loss, logits, probabilities).
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Ensure the model is in evaluation mode
        bnn.eval()

        # Update the model's internal state
        if len(bnn_history) > bnn.last_update_index:
            bnn.update_matrix(bnn_history, device=device)

        # Set the current index for the model
        bnn.current_index = len(bnn_history) - 1 if current_index is None else current_index
        bnn.bnn_history = bnn_history

        # Prepare dummy x_data and move ground_truth_labels to device
        x_data = torch.empty((1, bnn.input_size), device=device).fill_(-1)
        y_data = ground_truth_labels.to(device)

        with torch.no_grad():

            # Perform a forward pass to compute logits
            logits = bnn.forward_bce(x_data, device=device)

            # Apply sigmoid activation to get probabilities
            probabilities = torch.sigmoid(logits)

            # Compute BCE loss
            loss_fn = torch.nn.BCELoss()
            loss = loss_fn(probabilities, y_data)

        # Reset the current index
        bnn.current_index = None

        return loss, logits, probabilities



