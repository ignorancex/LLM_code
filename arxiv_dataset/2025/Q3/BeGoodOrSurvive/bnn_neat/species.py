"""Divides the population into species based on genomic distances."""
from itertools import count

from bnn_neat.config import ConfigParameter, DefaultClassConfig
from bnn_neat.math_util import mean, stdev


class Species(object):
    def __init__(self, key, generation):
        self.key = key
        self.created = generation
        self.last_improved = generation
        self.representative = None
        self.members = {}
        self.fitness = None
        self.adjusted_fitness = None
        self.fitness_history = []
        self.avg_fitness_history = []
        self.best_fitness_history = []
        self.stagnant = False


    def update(self, representative, members):
        self.representative = representative
        self.members = members

    def get_fitnesses(self):
        return [m.fitness for m in self.members.values()]

    def update_fitness_history(self, avg_fitness, best_fitness, stagnation_limit):
        """
        Updates the species fitness history for stagnation tracking.
        """
        self.species_avg_fitness_history.append(avg_fitness)
        self.species_best_fitness_history.append(best_fitness)

        # Limit history to stagnation limit
        if len(self.species_avg_fitness_history) > stagnation_limit:
            self.species_avg_fitness_history.pop(0)
        if len(self.species_best_fitness_history) > stagnation_limit:
            self.species_best_fitness_history.pop(0)

    def check_stagnation(self, stagnation_limit):
        """
        Checks if the species is stagnant by averaging fitness values over the stagnation window.
        """
        if len(self.species_avg_fitness_history) < stagnation_limit:
            return  # Not enough history to evaluate

        # Calculate average of the fitness window and compare to starting point
        avg_window = sum(self.species_avg_fitness_history) / len(self.species_avg_fitness_history)
        starting_avg = self.species_avg_fitness_history[0]
        avg_improvement = avg_window - starting_avg

        best_window = sum(self.species_best_fitness_history) / len(self.species_best_fitness_history)
        starting_best = self.species_best_fitness_history[0]
        best_improvement = best_window - starting_best

        # Mark as stagnant only if neither average nor best have improved
        if avg_improvement <= 0 and best_improvement <= 0:
            self.stagnant = True
            print(f"Species {self.key} marked stagnant. Avg: {avg_improvement:.4f}, Best: {best_improvement:.4f}")
        else:
            self.stagnant = False  # Reset if improvement is detected


class GenomeDistanceCache(object):
    def __init__(self, config):
        self.distances = {}
        self.config = config
        self.hits = 0
        self.misses = 0

    def __call__(self, genome0, genome1):
        g0 = genome0.key
        g1 = genome1.key
        d = self.distances.get((g0, g1))
        if d is None:
            # Distance is not already computed.
            d = genome0.distance(genome1, self.config)
            self.distances[g0, g1] = d
            self.distances[g1, g0] = d
            self.misses += 1
        else:
            self.hits += 1

        return d


class DefaultSpeciesSet(DefaultClassConfig):
    """ Encapsulates the default speciation scheme. """

    def __init__(self, config, reporters):
        # pylint: disable=super-init-not-called
        self.species_set_config = config
        self.reporters = reporters
        self.indexer = count(1)
        self.species = {}
        self.genome_to_species = {}

    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(param_dict,
                                  [ConfigParameter('compatibility_threshold', float)])

    def speciate(self, config, population, generation):
        """
        Place genomes into species by genetic similarity.

        Note that this method assumes the current representatives of the species are from the old
        generation, and that after speciation has been performed, the old representatives should be
        dropped and replaced with representatives from the new generation.  If you violate this
        assumption, you should make sure other necessary parts of the code are updated to reflect
        the new behavior.
        """
        assert isinstance(population, dict)

        compatibility_threshold = self.species_set_config.compatibility_threshold

        # Find the best representatives for each existing species.
        unspeciated = set(population)
        distances = GenomeDistanceCache(config.genome_config)
        new_representatives = {}
        new_members = {}
        for sid, s in self.species.items():
            candidates = []
            for gid in unspeciated:
                g = population[gid]
                d = distances(s.representative, g)
                candidates.append((d, g))

            # The new representative is the genome closest to the current representative.
            ignored_rdist, new_rep = min(candidates, key=lambda x: x[0])
            new_rid = new_rep.key
            new_representatives[sid] = new_rid
            new_members[sid] = [new_rid]
            unspeciated.remove(new_rid)

        # Partition population into species based on genetic similarity.
        while unspeciated:
            gid = unspeciated.pop()
            g = population[gid]

            # Find the species with the most similar representative.
            candidates = []
            for sid, rid in new_representatives.items():
                rep = population[rid]
                d = distances(rep, g)
                if d < compatibility_threshold:
                    candidates.append((d, sid))

            if candidates:
                ignored_sdist, sid = min(candidates, key=lambda x: x[0])
                new_members[sid].append(gid)
            else:
                # No species is similar enough, create a new species, using
                # this genome as its representative.
                sid = next(self.indexer)
                new_representatives[sid] = gid
                new_members[sid] = [gid]

        # Update species collection based on new speciation.
        self.genome_to_species = {}
        for sid, rid in new_representatives.items():
            s = self.species.get(sid)
            if s is None:
                s = Species(sid, generation)
                self.species[sid] = s

            members = new_members[sid]
            for gid in members:
                self.genome_to_species[gid] = sid

            member_dict = dict((gid, population[gid]) for gid in members)
            s.update(population[rid], member_dict)

        # Mean and std genetic distance info report
        if len(population) > 1:
            gdmean = mean(distances.distances.values())
            gdstdev = stdev(distances.distances.values())
            self.reporters.info(
                'Mean genetic distance {0:.3f}, standard deviation {1:.3f}'.format(gdmean, gdstdev))

    def get_species_id(self, individual_id):
        return self.genome_to_species[individual_id]

    def get_species(self, individual_id):
        sid = self.genome_to_species[individual_id]
        return self.species[sid]
