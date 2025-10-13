"""Implements the core evolution algorithm."""
from utils.rates_utils import get_initial_rates, get_final_rates, print_config_values
from bnn_neat.math_util import mean
from bnn_neat.reporting import ReporterSet
from bnn_neat.genome import DefaultGenome
import json
import datetime
from mpi4py import MPI
import pickle
import json
import os
import torch
class CompleteExtinctionException(Exception):
    pass


class Population(object):
    """
    This class implements the core evolution algorithm:
        1. Evaluate fitness of all genomes.
        2. Check to see if the termination criterion is satisfied; exit if it is.
        3. Generate the next generation from the current population.
        4. Partition the new generation into species based on genetic similarity.
        5. Go to 1.
    """

    def __init__(self, config, initial_state=None):
        self.reporters = ReporterSet()
        self.config = config
        print_config_values(self.config)
        stagnation = config.stagnation_type(config.stagnation_config, self.reporters)
        self.reproduction = config.reproduction_type(config.reproduction_config,
                                                     self.reporters,
                                                     stagnation)
        if config.fitness_criterion == 'max':
            self.fitness_criterion = max
        elif config.fitness_criterion == 'min':
            self.fitness_criterion = min
        elif config.fitness_criterion == 'mean':
            self.fitness_criterion = mean
        elif not config.no_fitness_termination:
            raise RuntimeError(
                "Unexpected fitness_criterion: {0!r}".format(config.fitness_criterion))

        if initial_state is None:
            # Create a population from scratch, then partition into species.
            self.population = self.reproduction.create_new(config.genome_type,
                                                           config.genome_config,
                                                           config.pop_size)
            self.species = config.species_set_type(config.species_set_config, self.reporters)
            self.generation = 0
            self.species.speciate(config, self.population, self.generation)
        else:
            self.population, self.species, self.generation = initial_state

        self.best_genome = None
        self.previous_best_genome = None

        self.champion_architectures = {}

        self.fitness_improvement_history = []  # To track recent fitness improvements


        self.neat_iteration = None

        self.save_dir = None

    def add_reporter(self, reporter):
        self.reporters.add(reporter)

    def remove_reporter(self, reporter):
        self.reporters.remove(reporter)

    def adjust_compatibility_threshold(self, config, species_count):
        """
        Adjusts the compatibility threshold to maintain the desired number of species.
        This version is improved with a moving average of species count to avoid reacting to single-generation spikes.
        """
        desired_min_species = 8
        desired_max_species = 16
        adjustment_step = 0.1  # Gradual, precise adjustment
        moving_average_length = 5  # Moving average over the last 5 generations

        # Initialize species count history if not present
        if not hasattr(self, "species_count_history"):
            self.species_count_history = []

        # Track species count history for moving average
        self.species_count_history.append(species_count)
        if len(self.species_count_history) > moving_average_length:
            self.species_count_history.pop(0)

        # Calculate the moving average of species count
        avg_species_count = sum(self.species_count_history) / len(self.species_count_history)
        print(f"Generation {self.generation}: Avg Species Count (Last {moving_average_length} Gens): {avg_species_count:.2f}")

        # Only adjust the threshold if the moving average is consistently off target
        if avg_species_count < desired_min_species:
            config.species_set_config.compatibility_threshold -= adjustment_step
            print(f"Decreasing compatibility threshold to {config.species_set_config.compatibility_threshold:.2f} (Species: {species_count})")

        elif avg_species_count > desired_max_species:
            config.species_set_config.compatibility_threshold += adjustment_step
            print(f"Increasing compatibility threshold to {config.species_set_config.compatibility_threshold:.2f} (Species: {species_count})")

        # Ensure the threshold stays within a reasonable range for stability
        min_threshold = 5.0
        max_threshold = 20.0
        config.species_set_config.compatibility_threshold = max(
            min_threshold, 
            min(max_threshold, config.species_set_config.compatibility_threshold)
        )

        # Clear, detailed logging for tracking
        print(f"Generation {self.generation}: Adjusted Compatibility Threshold = {config.species_set_config.compatibility_threshold:.2f}")

    def freeze_top_genomes(self):
        """
        Mark the top 5% of genomes as frozen (no_mutation) based on fitness.
        """
        # Sort genomes by fitness (descending)
        sorted_genomes = sorted(self.population.values(), key=lambda g: g.fitness, reverse=True)

        # Determine the top 5% cutoff
        top_5_count = max(1, int(0.05 * len(sorted_genomes)))  # At least 1 genome

        # Mark the top genomes as frozen
        for genome in sorted_genomes[:top_5_count]:
            genome.no_mutation = True  # Add a freezing flag

        print(f"Generation {self.generation}: {top_5_count} genomes frozen from mutation.")


    def adjust_mutation_rates(self, factor=1.1, decrease_factor=0.85, base_rate=0.9, final_rate=0.1, max_gens=25, stability_factor=0.9):
        """
        Dynamically adjust mutation rates based on fitness improvement and species diversity.
        Additionally, every 5 generations, reduce all mutation rates by 15% to stabilize evolution.
        
        - `factor`: Multiplier for increasing mutation rates (default 1.1).
        - `decrease_factor`: Multiplier for decreasing mutation rates (default 0.9).
        - `base_rate`: Initial mutation rate for tapering (default 0.9).
        - `final_rate`: Final mutation rate for tapering (default 0.1).
        - `stability_factor`: The percentage by which all rates are reduced every 5 generations (default 0.85).
        """
        mutation_rates = [
            "bias_mu_mutate_rate",
            "bias_mu_replace_rate",
            "bias_sigma_mutate_rate",
            "bias_sigma_replace_rate",
            "weight_mu_mutate_rate",
            "weight_mu_replace_rate",
            "weight_sigma_mutate_rate",
            "weight_sigma_replace_rate",
            "response_mu_mutate_rate",
            "response_mu_replace_rate",
            "response_sigma_mutate_rate",
            "response_sigma_replace_rate",
            "node_add_prob",
            "node_delete_prob",
            "conn_add_prob",
            "conn_delete_prob",
        ]
        moving_average_length = 5  # Can be adjusted for smoother or faster response
        mutation_adjustment_threshold = 10


        # Compute fitness improvement
        if self.best_genome is None or self.previous_best_genome is None:
            fitness_improvement = 1  # Default if no valid genomes exist
        else:
            fitness_improvement = max(0, self.best_genome.fitness - self.previous_best_genome.fitness)

        # Track fitness improvement using a moving average
        self.fitness_improvement_history.append(fitness_improvement)
        if len(self.fitness_improvement_history) > moving_average_length:  # Last n generations
            self.fitness_improvement_history.pop(0)

        if len(self.fitness_improvement_history) > 0:
            avg_improvement = sum(self.fitness_improvement_history) / len(self.fitness_improvement_history)
        else:
            avg_improvement = 0

        print(f"Average Fitness Improvement (Last 5 Generations): {avg_improvement}")

        species_count = len(self.species.species)
        max_generations = max_gens if max_gens else 100
        progress = self.generation / max_generations

        for rate in mutation_rates:
            current_rate = getattr(self.config.genome_config, rate, None)
            if current_rate is not None:
                # Standard dynamic mutation rate adjustments
                if avg_improvement < 0.02 and self.generation > mutation_adjustment_threshold:  # If fitness is stagnating
                    new_rate = min(current_rate * factor, 1.0)  # Increase mutation (cap at 1.0)
                elif avg_improvement > 0.05:  # If fitness is steadily improving
                    new_rate = max(current_rate * decrease_factor, 0.01)  # Decrease mutation (floor at 0.01)
                else:
                    # Use tapered mutation rate if no significant fitness change
                    taper_rate = base_rate + progress * (final_rate - base_rate)
                    new_rate = max(current_rate, taper_rate)  # Ensure no reduction below taper_rate

                # Apply a 15% reduction to all mutation rates every 5 generations
                new_rate = max(0.9 - ((self.generation / max_gens) * 0.8), 0.01)

                setattr(self.config.genome_config, rate, new_rate)  # Update mutation rate

        # Log the mutation rate adjustment for debugging
        print(
            f"Generation {self.generation}: Adjusted mutation rates based on fitness improvement "
            f"({fitness_improvement:.4f}) and species count ({species_count})."
        )

        # Print stability enforcement every 5 generations
        if self.generation % mutation_adjustment_threshold == 0:
            print(f"Generation {self.generation}: Stability mechanism triggered. Mutation rates reduced by 15%.")

        adjusted_rates = {rate: getattr(self.config.genome_config, rate) for rate in mutation_rates}
        print(f"Generation {self.generation}: Mutation rates: {adjusted_rates}")


    def set_stagnation_limit(self, new_limit):
        """
        Sets a new stagnation limit in the configuration.
        """
        self.config.stagnation_config.species_fitness_func_args['max_stagnation'] = new_limit
        print(f"Set stagnation limit to {new_limit}")

    def save_all_genomes(self, attention_layers, rank):
        """ Saves all genomes in the population, overwriting previous versions if they exist. """
        if rank == 0:
            os.makedirs(self.save_dir, exist_ok=True)  # Ensure directory exists

            for genome_id, genome in self.population.items():
                save_path = os.path.join(self.save_dir, f"genome_{genome_id}.pth")

                torch.save({
                    'genome': genome,
                    'config': self.config  # Store NEAT config for later reconstruction
                }, save_path)

                print(f"Saved (or updated) genome {genome_id} to {save_path}")

            # ✅ Save attention layers separately since they're the same for all genomes
            attention_save_path = os.path.join(self.save_dir, f"attention_layers_{self.neat_iteration}.pth")
            torch.save({'attention_layers': attention_layers}, attention_save_path)
            print(f"Saved attention layers to {attention_save_path}")


    def run(self, fitness_function, n=None, neat_iteration="NoneSet", comm=None, max_gens=None, attention_layers=None):
        rank = comm.Get_rank()
        size = comm.Get_size()

        if self.neat_iteration is None:
            self.neat_iteration = neat_iteration

        if self.config.no_fitness_termination and (n is None):
            raise RuntimeError("Cannot have no generational limit with no fitness termination")

        evolution_data = {"generations": []}
        k = 0

        while n is None or k < n:
            current_time = datetime.datetime.now()
            print(current_time)
            k += 1

            # Broadcast compatibility threshold and stagnation limit at the beginning
            compatibility_threshold = comm.bcast(self.config.species_set_config.compatibility_threshold if rank == 0 else None, root=0)
            self.config.species_set_config.compatibility_threshold = compatibility_threshold

            stagnation_limit = comm.bcast(self.config.stagnation_config.max_stagnation if rank == 0 else None, root=0)
            self.config.stagnation_config.max_stagnation = stagnation_limit

            comm.Barrier()  # Ensure all ranks have the same values before proceeding


            self.reporters.start_generation(self.generation)

            # Calculate average and best fitness for each species and update history
            """
            if rank == 0:
                for species in self.species.species.values():
                    avg_fitness = sum([genome.fitness for genome in species.members.values()]) / len(species.members)
                    best_fitness = max([genome.fitness for genome in species.members.values()])

                    # Initialize history if it does not exist
                    if not hasattr(species, "avg_fitness_history"):
                        species.avg_fitness_history = []
                    if not hasattr(species, "best_fitness_history"):
                        species.best_fitness_history = []

                    # Update the fitness history
                    species.avg_fitness_history.append(avg_fitness)
                    species.best_fitness_history.append(best_fitness)

                    # Limit history to the stagnation limit
                    if len(species.avg_fitness_history) > self.config.stagnation_config.max_stagnation:
                        species.avg_fitness_history.pop(0)
                    if len(species.best_fitness_history) > self.config.stagnation_config.max_stagnation:
                        species.best_fitness_history.pop(0)
            """



            # Check for dynamic stagnation adjustment
            if rank == 0:  # Only the main rank handles this adjustment
                current_stagnation = self.config.stagnation_config.max_stagnation

                if self.generation % 10 == 0 and self.generation != 0:  # Adjust every 5 generations instead of waiting too long
                    current_stagnation = self.config.stagnation_config.max_stagnation

                    print(f"Generation {self.generation}: Adjusting mutation rates.")
                    self.adjust_mutation_rates(max_gens=max_gens)

                    """
                    if current_stagnation > 6:
                        self.config.stagnation_config.max_stagnation -= 1  # Gradual decay
                    elif current_stagnation > 3:
                        self.config.stagnation_config.max_stagnation -= 0.5  # Slower decay when stagnation is low

                    print(f"Generation {self.generation}: Adjusted stagnation to {self.config.stagnation_config.max_stagnation}")
                    """

                # Broadcast the updated stagnation limit and compatibility threshold
                stagnation_limit = comm.bcast(self.config.stagnation_config.max_stagnation if rank == 0 else 0.0, root=0)
                self.config.stagnation_config.max_stagnation = stagnation_limit

            comm.Barrier()
            
            # All ranks execute the fitness function
            should_stop = fitness_function(list(self.population.items()), self.config, k)


            if should_stop:
                if rank == 0:
                    print("Stopping evolution: Fitness function requested early stopping.")
                break

            # ✅ Ensure save directory exists before saving genomes
            if rank == 0:
                self.save_dir = f"saved_genomes/neat_iteration_{self.neat_iteration}"
                os.makedirs(self.save_dir, exist_ok=True)

            comm.Barrier()

            self.save_all_genomes(attention_layers, rank)
            

            # Rank 0 handles reporting and statistics
            if rank == 0:
                # Gather and report statistics.
                best = None
                generation_data = {
                    "generation": self.generation,
                    "population_size": len(self.population),
                    "species_count": len(self.species.species),
                    "species_details": [],
                    "best_genome": None, 
                    "previous_best_genome": None,
                    "compatibility_thres": self.config.species_set_config.compatibility_threshold,
                    "stagnation_limit": self.config.stagnation_config.max_stagnation
                }

                for g in self.population.values():
                    if g.fitness is None:
                        raise RuntimeError(f"Fitness not assigned to genome {g.key}")

                    if best is None: 
                        best = g
                    
                    elif g.fitness > best.fitness:
                        best = g

                # Collect species details
                for sid, species in self.species.species.items():
                    print("species details")
                    # Calculate average ethical score for the species
                    genome_ids = list(species.members.keys())
                    ethical_scores = []

                    for g_id in genome_ids:
                        if g_id in self.population:
                            ethical_score_history = self.population[g_id].ethical_score_history
                            # Extract only the scores from (idx, score) tuples
                            scores = [score for _, score in ethical_score_history]
                            if scores:  # Only calculate the average if there are scores
                                ethical_scores.append(sum(scores) / len(scores))  # Average of this genome's ethical scores

                    # Calculate the average ethical score for the species
                    avg_ethical_score = sum(ethical_scores) / len(ethical_scores) if ethical_scores else None

                    species_data = {
                        "species_id": sid,
                        "age": self.generation - species.created,
                        "size": len(species.members),
                        "fitness": species.fitness,
                        "adjusted_fitness": species.adjusted_fitness,
                        "stagnation": self.generation - species.last_improved,
                        "genome_ids": genome_ids,  # Add genome IDs in this species
                        "avg_ethical_score": avg_ethical_score  # Add aggregated ethical score
                    }
                    generation_data["species_details"].append(species_data)


                # Add the best genome of this generation
                if best is not None:
                    generation_data["best_genome"] = {
                        "key": best.key,
                        "fitness": best.fitness,
                        "size": best.size()
                    }
                    self.champion_architectures[self.generation] = best.architecture

                if self.previous_best_genome is not None:
                    generation_data["previous_best_genome"] = {
                        "key": self.previous_best_genome.key,
                        "fitness": self.previous_best_genome.fitness,
                        "size": self.previous_best_genome.size(),
                    }

                # Append generation data
                evolution_data["generations"].append(generation_data)


                # Reporters handle the post-evaluation
                self.reporters.post_evaluate(self.config, self.population, self.species, best)

                # Track the best genome ever seen.
                if self.best_genome is None:
                    self.best_genome = best
                    self.previous_best_genome = best  # Initialize previous best as well
                elif best.fitness > self.best_genome.fitness:
                    self.previous_best_genome = self.best_genome
                    self.best_genome = best


                if not self.config.no_fitness_termination:
                    # End if the fitness threshold is reached.
                    fv = self.fitness_criterion(g.fitness for g in self.population.values())
                    if fv >= self.config.fitness_threshold:
                        self.reporters.found_solution(self.config, self.generation, best)
                        break


                # Create the next generation from the current generation.
                
                self.population = self.reproduction.reproduce(
                    self.config, self.species, self.config.pop_size, self.generation
                )

                # Check for complete extinction.
                if not self.species.species:
                    self.reporters.complete_extinction()

                    # If requested by the user, create a completely new population,
                    # otherwise raise an exception.
                    if self.config.reset_on_extinction:
                        self.population = self.reproduction.create_new(
                            self.config.genome_type,
                            self.config.genome_config,
                            self.config.pop_size
                        )
                    else:
                        raise CompleteExtinctionException()

                # Divide the new population into species.
                self.species.speciate(self.config, self.population, self.generation)

                if rank == 0:
                    species_count = len(self.species.species)
                    desired_min_species = 8
                    desired_max_species = 16
                    self.adjust_compatibility_threshold(
                        config=self.config,
                        species_count=species_count,
                    )


                if rank == 0:
                    self.reporters.end_generation(self.config, self.population, self.species)

                self.generation += 1

        # At the end of evolution, rank 0 saves the results
        if rank == 0:
            if self.config.no_fitness_termination:
                self.reporters.found_solution(self.config, self.generation, self.best_genome)

            # Save the evolution data to a JSON file
            with open(f"recycle_prod_evolution_generation_data_{neat_iteration}.json", "w") as json_file:
                json.dump(evolution_data, json_file, indent=4)

            print(f"Saved evo gen data to recycle_prod_evolution_generation_data_{neat_iteration}.json")

        if rank == 0:
            architecture_file = f"recycle_champion_architectures_{neat_iteration}.json"
            with open(architecture_file, "w") as json_file:
                json.dump(self.champion_architectures, json_file, indent=4)

            print(f"Saved champion architectures to {architecture_file}")


        # If Rank 0, send the serialized best genome to all other ranks
        if rank == 0:
            serialized_best_genome = pickle.dumps(self.best_genome)
            for r in range(1, size):
                comm.send(serialized_best_genome, dest=r, tag=200)
        else:
            # Each other rank receives the serialized genome
            serialized_best_genome = comm.recv(source=0, tag=200)

        # Deserialize the received genome
        self.best_genome = pickle.loads(serialized_best_genome)
        return self.best_genome

