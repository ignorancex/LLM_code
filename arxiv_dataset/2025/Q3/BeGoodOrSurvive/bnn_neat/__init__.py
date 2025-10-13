"""A NEAT (NeuroEvolution of Augmenting Topologies) implementation"""
import bnn_neat.nn as nn
import bnn_neat.ctrnn as ctrnn
import bnn_neat.iznn as iznn
import bnn_neat.distributed as distributed

from bnn_neat.config import Config
from bnn_neat.population import Population, CompleteExtinctionException
from bnn_neat.genome import DefaultGenome
from bnn_neat.reproduction import DefaultReproduction
from bnn_neat.stagnation import DefaultStagnation
from bnn_neat.reporting import StdOutReporter
from bnn_neat.species import DefaultSpeciesSet
from bnn_neat.statistics import StatisticsReporter
from bnn_neat.parallel import ParallelEvaluator
from bnn_neat.distributed import DistributedEvaluator, host_is_local
from bnn_neat.threaded import ThreadedEvaluator
from bnn_neat.checkpoint import Checkpointer
from bnn_neat.genes import DefaultConnectionGene, DefaultNodeGene

