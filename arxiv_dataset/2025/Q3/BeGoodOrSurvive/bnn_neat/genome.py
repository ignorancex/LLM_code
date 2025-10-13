"""Handles genomes (individuals in the population)."""
import copy
import sys
from itertools import count
from random import choice, random, shuffle
import random as rndm

from bnn_neat.activations import ActivationFunctionSet
from bnn_neat.aggregations import AggregationFunctionSet
from bnn_neat.config import ConfigParameter, write_pretty_params
from bnn_neat.genes import DefaultConnectionGene, DefaultNodeGene
from bnn_neat.graphs import creates_cycle
from bnn_neat.graphs import required_for_output


class DefaultGenomeConfig(object):
    """Sets up and holds configuration information for the DefaultGenome class."""
    allowed_connectivity = ['unconnected', 'fs_neat_nohidden', 'fs_neat', 'fs_neat_hidden',
                            'full_nodirect', 'full', 'full_direct',
                            'partial_nodirect', 'partial', 'partial_direct']

    def __init__(self, params):
        # Set default values for node_gene_type and connection_gene_type if not in params
        self.node_gene_type = params.get('node_gene_type', DefaultNodeGene)
        self.connection_gene_type = params.get('connection_gene_type', DefaultConnectionGene)

        # Create full set of available activation functions.
        self.activation_defs = ActivationFunctionSet()
        # ditto for aggregation functions - name difference for backward compatibility
        self.aggregation_function_defs = AggregationFunctionSet()
        self.aggregation_defs = self.aggregation_function_defs

        self._params = [ConfigParameter('num_inputs', int),
                        ConfigParameter('num_outputs', int),
                        ConfigParameter('num_hidden', int),
                        ConfigParameter('feed_forward', bool),
                        ConfigParameter('compatibility_disjoint_coefficient', float),
                        ConfigParameter('compatibility_weight_coefficient', float),
                        ConfigParameter('conn_add_prob', float),
                        ConfigParameter('conn_delete_prob', float),
                        ConfigParameter('node_add_prob', float),
                        ConfigParameter('node_delete_prob', float),
                        ConfigParameter('single_structural_mutation', bool, 'false'),
                        ConfigParameter('structural_mutation_surer', str, 'default'),
                        ConfigParameter('initial_connection', str, 'unconnected'),

                        ConfigParameter('bias_mu_init_mean', float, 0.0),
                        ConfigParameter('bias_mu_init_stdev', float, 1.0),
                        ConfigParameter('bias_mu_min_value', float, -3.0),
                        ConfigParameter('bias_mu_max_value', float, 3.0),
                        ConfigParameter('bias_mu_replace_rate', float, 0.1),

                        ConfigParameter('bias_sigma_init_mean', float, 1.0),
                        ConfigParameter('bias_sigma_init_stdev', float, 0.1),
                        ConfigParameter('bias_sigma_min_value', float, 0.01),
                        ConfigParameter('bias_sigma_max_value', float, 2.0),
                        ConfigParameter('bias_sigma_replace_rate', float, 0.1),

                        ConfigParameter('response_mu_init_mean', float, 1.0),
                        ConfigParameter('response_mu_init_stdev', float, 0.1),
                        ConfigParameter('response_mu_min_value', float, 0.1),
                        ConfigParameter('response_mu_max_value', float, 5.0),
                        ConfigParameter('response_mu_replace_rate', float, 0.3),

                        ConfigParameter('response_sigma_init_mean', float, 0.1),
                        ConfigParameter('response_sigma_init_stdev', float, 0.05),
                        ConfigParameter('response_sigma_min_value', float, 0.01),
                        ConfigParameter('response_sigma_max_value', float, 1.0),
                        ConfigParameter('response_sigma_replace_rate', float, 0.3),

                        ConfigParameter('weight_mu_init_mean', float, 0.0),
                        ConfigParameter('weight_mu_init_stdev', float, 1.0),
                        ConfigParameter('weight_sigma_init_mean', float, 1.0),
                        ConfigParameter('weight_sigma_init_stdev', float, 0.1),

                         # Added config for 'enabled'
                        ConfigParameter('enabled_default', bool, True),  # Default for 'enabled' flag
                        ConfigParameter('enabled_mutate_rate', float, 0.1)  # Mutation rate for 'enabled'

                        ]

        # Gather configuration data from the gene classes.
        self.node_gene_type = params['node_gene_type']
        self._params += self.node_gene_type.get_config_params()
        self.connection_gene_type = params['connection_gene_type']
        self._params += self.connection_gene_type.get_config_params()

        # Use the configuration data to interpret the supplied parameters.
        for p in self._params:
            setattr(self, p.name, p.interpret(params))

        self.node_gene_type.validate_attributes(self)
        self.connection_gene_type.validate_attributes(self)

        # By convention, input pins have negative keys, and the output
        # pins have keys 0,1,...
        self.input_keys = [-i - 1 for i in range(self.num_inputs)]
        self.output_keys = [i for i in range(self.num_outputs)]

        self.connection_fraction = None

        # Verify that initial connection type is valid.
        # pylint: disable=access-member-before-definition
        if 'partial' in self.initial_connection:
            c, p = self.initial_connection.split()
            self.initial_connection = c
            self.connection_fraction = float(p)
            if not (0 <= self.connection_fraction <= 1):
                raise RuntimeError(
                    "'partial' connection value must be between 0.0 and 1.0, inclusive.")

        assert self.initial_connection in self.allowed_connectivity

        # Verify structural_mutation_surer is valid.
        # pylint: disable=access-member-before-definition
        if self.structural_mutation_surer.lower() in ['1', 'yes', 'true', 'on']:
            self.structural_mutation_surer = 'true'
        elif self.structural_mutation_surer.lower() in ['0', 'no', 'false', 'off']:
            self.structural_mutation_surer = 'false'
        elif self.structural_mutation_surer.lower() == 'default':
            self.structural_mutation_surer = 'default'
        else:
            error_string = f"Invalid structural_mutation_surer {self.structural_mutation_surer!r}"
            raise RuntimeError(error_string)

        self.node_indexer = None


        # Validate attributes as necessary
        self.node_gene_type.validate_attributes(self)
        self.connection_gene_type.validate_attributes(self)

        self.mutation_history = []

    def add_activation(self, name, func):
        self.activation_defs.add(name, func)

    def add_aggregation(self, name, func):
        self.aggregation_function_defs.add(name, func)

    def save(self, f):
        if 'partial' in self.initial_connection:
            if not (0 <= self.connection_fraction <= 1):
                raise RuntimeError(
                    "'partial' connection value must be between 0.0 and 1.0, inclusive.")
            f.write(f'initial_connection      = {self.initial_connection} {self.connection_fraction}\n')
        else:
            f.write(f'initial_connection      = {self.initial_connection}\n')

        assert self.initial_connection in self.allowed_connectivity

        write_pretty_params(f, self, [p for p in self._params
                                      if 'initial_connection' not in p.name])


    def get_new_node_key(self, node_dict):
        if not hasattr(self, 'node_indexer') or self.node_indexer is None:
            if node_dict:
                existing_ids = set(node_dict.keys())
                # Exclude input and output node IDs
                max_existing_id = max([nid for nid in existing_ids if nid >= 0], default=-1)
                self.node_indexer = count(max_existing_id + 1)
            else:
                self.node_indexer = count(0)

        new_id = next(self.node_indexer)

        while new_id in node_dict:
            new_id = next(self.node_indexer)

        assert new_id not in node_dict, f"Generated node ID {new_id} already exists."

        return new_id

    def check_structural_mutation_surer(self):
        if self.structural_mutation_surer == 'true':
            return True
        elif self.structural_mutation_surer == 'false':
            return False
        elif self.structural_mutation_surer == 'default':
            return self.single_structural_mutation
        else:
            error_string = f"Invalid structural_mutation_surer {self.structural_mutation_surer!r}"
            raise RuntimeError(error_string)

    def to_dict(self):
        """
        Converts the configuration parameters to a dictionary.
        """
        return {
            "num_inputs": self.num_inputs,
            "num_outputs": self.num_outputs,
            "num_hidden": self.num_hidden,
            "feed_forward": self.feed_forward,
            "compatibility_disjoint_coefficient": self.compatibility_disjoint_coefficient,
            "compatibility_weight_coefficient": self.compatibility_weight_coefficient,
            "conn_add_prob": self.conn_add_prob,
            "conn_delete_prob": self.conn_delete_prob,
            "node_add_prob": self.node_add_prob,
            "node_delete_prob": self.node_delete_prob,
            "single_structural_mutation": self.single_structural_mutation,
            "structural_mutation_surer": self.structural_mutation_surer,
            "initial_connection": self.initial_connection,
            "bias_mu_init_mean": self.bias_mu_init_mean,
            "bias_mu_init_stdev": self.bias_mu_init_stdev,
            "bias_mu_min_value": self.bias_mu_min_value,
            "bias_mu_max_value": self.bias_mu_max_value,
            "bias_mu_replace_rate": self.bias_mu_replace_rate,
            "bias_sigma_init_mean": self.bias_sigma_init_mean,
            "bias_sigma_init_stdev": self.bias_sigma_init_stdev,
            "bias_sigma_min_value": self.bias_sigma_min_value,
            "bias_sigma_max_value": self.bias_sigma_max_value,
            "bias_sigma_replace_rate": self.bias_sigma_replace_rate,
            "response_mu_init_mean": self.response_mu_init_mean,
            "response_mu_init_stdev": self.response_mu_init_stdev,
            "response_mu_min_value": self.response_mu_min_value,
            "response_mu_max_value": self.response_mu_max_value,
            "response_mu_replace_rate": self.response_mu_replace_rate,
            "response_sigma_init_mean": self.response_sigma_init_mean,
            "response_sigma_init_stdev": self.response_sigma_init_stdev,
            "response_sigma_min_value": self.response_sigma_min_value,
            "response_sigma_max_value": self.response_sigma_max_value,
            "response_sigma_replace_rate": self.response_sigma_replace_rate,
            "weight_mu_init_mean": self.weight_mu_init_mean,
            "weight_mu_init_stdev": self.weight_mu_init_stdev,
            "weight_sigma_init_mean": self.weight_sigma_init_mean,
            "weight_sigma_init_stdev": self.weight_sigma_init_stdev,
            "enabled_default": self.enabled_default,
            "enabled_mutate_rate": self.enabled_mutate_rate
        }


class DefaultGenome(object):
    """
    A genome for generalized neural networks.

    Terminology
        pin: Point at which the network is conceptually connected to the external world;
             pins are either input or output.
        node: Analog of a physical neuron.
        connection: Connection between a pin/node output and a node's input, or between a node's
             output and a pin/node input.
        key: Identifier for an object, unique within the set of similar objects.

    Design assumptions and conventions.
        1. Each output pin is connected only to the output of its own unique
           neuron by an implicit connection with weight one. This connection
           is permanently enabled.
        2. The output pin's key is always the same as the key for its
           associated neuron.
        3. Output neurons can be modified but not deleted.
        4. The input values are applied to the input pins unmodified. ##############################################################################################################################
    """

    @classmethod
    def parse_config(cls, param_dict):
        param_dict['node_gene_type'] = DefaultNodeGene
        param_dict['connection_gene_type'] = DefaultConnectionGene
        return DefaultGenomeConfig(param_dict)

    @classmethod
    def write_config(cls, f, config):
        config.save(f)

    def __init__(self, key):
        # Unique identifier for a genome instance.
        self.key = key
        self.mutation_history = []

        self.evaluation_window = 5

        # (gene_key, gene) pairs for gene sets.
        self.connections = {}
        self.nodes = {}

        # Fitness results.
        self.fitness = None
        self.fitness_history = []
        self.parent_fitness = None

        self.parents = []  # Stores genome IDs of parents

        self.architecture = None

        # Add histories for decisions and ethical scores
        self.decision_history = []
        self.ethical_score_history = []

        self.device = None

    def update_fitness_history(self, fitness):
        """ Add current fitness to species history for stagnation checks """
        self.fitness_history.append(fitness)

    def configure_new(self, config):
        """Configure a new genome based on the given configuration."""

        # Create node genes for the output nodes.
        for node_key in config.output_keys:
            node_gene = self.create_node(config, node_key)
            # Set properties specific to output nodes if necessary
            # Ensure output nodes have the desired activation function
            node_gene.activation = 'sigmoid'  # or your desired activation function for outputs
            self.nodes[node_key] = node_gene

        # Add hidden nodes if requested.
        if config.num_hidden > 0:
            for i in range(config.num_hidden):
                node_key = config.get_new_node_key(self.nodes)
                assert node_key not in self.nodes
                node = self.create_node(config, node_key)
                self.nodes[node_key] = node

        # Add connections based on initial connectivity type.

        if 'fs_neat' in config.initial_connection:
            if config.initial_connection == 'fs_neat_nohidden':
                self.connect_fs_neat_nohidden(config)
            elif config.initial_connection == 'fs_neat_hidden':
                self.connect_fs_neat_hidden(config)
            else:
                if config.num_hidden > 0:
                    print(
                        "Warning: initial_connection = fs_neat will not connect to hidden nodes;",
                        "\tif this is desired, set initial_connection = fs_neat_nohidden;",
                        "\tif not, set initial_connection = fs_neat_hidden",
                        sep='\n', file=sys.stderr)
                self.connect_fs_neat_nohidden(config)
        elif 'full' in config.initial_connection:
            if config.initial_connection == 'full_nodirect':
                self.connect_full_nodirect(config)
            elif config.initial_connection == 'full_direct':
                self.connect_full_direct(config)
            else:
                if config.num_hidden > 0:
                    print(
                        "Warning: initial_connection = full with hidden nodes will not do direct input-output connections;",
                        "\tif this is desired, set initial_connection = full_nodirect;",
                        "\tif not, set initial_connection = full_direct",
                        sep='\n', file=sys.stderr)
                self.connect_full_nodirect(config)
        elif 'partial' in config.initial_connection:
            if config.initial_connection == 'partial_nodirect':
                self.connect_partial_nodirect(config)
            elif config.initial_connection == 'partial_direct':
                self.connect_partial_direct(config)
            else:
                if config.num_hidden > 0:
                    print(
                        "Warning: initial_connection = partial with hidden nodes will not do direct input-output connections;",
                        f"\tif this is desired, set initial_connection = partial_nodirect {config.connection_fraction};",
                        f"\tif not, set initial_connection = partial_direct {config.connection_fraction}",
                        sep='\n', file=sys.stderr)
                self.connect_partial_nodirect(config)
    '''
    def configure_crossover(self, genome1, genome2, config):
        # Determine the more fit parent
        if genome1.fitness > genome2.fitness:
            parent1, parent2 = genome1, genome2
        elif genome1.fitness < genome2.fitness:
            parent1, parent2 = genome2, genome1
        else:
            # If equal fitness, randomly choose parent1
            parent1, parent2 = (genome1, genome2) if rndm.random() > 0.5 else (genome2, genome1)

        parent1_id = parent1.key
        parent2_id = parent2.key

        # Collect all nodes from both parents
        parent1_nodes = parent1.nodes
        parent2_nodes = parent2.nodes
        input_node_ids = set(config.input_keys)
        output_node_ids = set(config.output_keys)
        all_node_ids = set(parent1_nodes.keys()).union(parent2_nodes.keys()).union(input_node_ids).union(output_node_ids)
        print("all_node_ids", all_node_ids)

        # Initialize self.nodes with nodes from both parents
        self.nodes = {}
        for node_id in all_node_ids:
            node1 = parent1_nodes.get(node_id)
            node2 = parent2_nodes.get(node_id)

            if node1 and node2:
                # Matching node: crossover attributes
                new_node = node1.crossover(node2, parent1_id, parent2_id, output_node_ids)
                self.nodes[node_id] = new_node
            elif node1:
                # Node only in fitter parent
                copied_node = node1.copy()
                copied_node.parent_info = {'inherited_from': parent1_id}
                self.nodes[node_id] = copied_node
            elif node2:
                # Node only in less fit parent
                # Include if parents have equal fitness or if the node is connected in the child genome
                if parent1.fitness == parent2.fitness:
                    copied_node = node2.copy()
                    copied_node.parent_info = {'inherited_from': parent2_id}
                    self.nodes[node_id] = copied_node

        print("self.nodes", self.nodes)

        # Build dictionaries of connections indexed by innovation number
        parent1_connections = {conn.innovation_number: conn for conn in parent1.connections.values()}
        parent2_connections = {conn.innovation_number: conn for conn in parent2.connections.values()}
        all_innovations = set(parent1_connections.keys()).union(parent2_connections.keys())

        # Include connections with parent info
        self.connections = {}
        for innov_num in all_innovations:
            conn1 = parent1_connections.get(innov_num)
            conn2 = parent2_connections.get(innov_num)

            if conn1 and conn2:
                # Matching gene: crossover attributes
                new_conn = conn1.crossover(conn2, parent1_id, parent2_id)
                self.connections[new_conn.key] = new_conn
            elif conn1:
                # Gene only in fitter parent
                copied_conn = conn1.copy()
                copied_conn.parent_info = {'inherited_from': parent1_id}
                self.connections[copied_conn.key] = copied_conn
            elif conn2:
                # Gene only in less fit parent
                # Include if both nodes exist in the child genome
                if conn2.key[0] in self.nodes and conn2.key[1] in self.nodes:
                    copied_conn = conn2.copy()
                    copied_conn.parent_info = {'inherited_from': parent2_id}
                    self.connections[copied_conn.key] = copied_conn

        # Optionally prune any nodes that are not connected
        #self.prune_unused_genes(config)

        # Print parentage for verification
        print(f"\n=== Parentage After Crossover for Genome {self.key} ===")
        print("Nodes:")
        for node_id, node in self.nodes.items():
            if hasattr(node, 'parent_info'):
                if 'attributes' in node.parent_info:
                    print(f"Node {node_id}: Inherited attributes from parents {parent1_id} and {parent2_id}")
                    for attr_name, info in node.parent_info['attributes'].items():
                        print(f"  Attribute '{attr_name}' inherited from parent {info['inherited_from']}: {info['value']}")
                else:
                    print(f"Node {node_id}: Inherited entirely from parent {node.parent_info['inherited_from']}")
            else:
                print(f"Node {node_id}: No parent info available")

        print("Connections:")
        for (in_node, out_node), conn in self.connections.items():
            if hasattr(conn, 'parent_info'):
                if 'attributes' in conn.parent_info:
                    print(f"Connection {in_node} -> {out_node}: Inherited attributes from parents {parent1_id} and {parent2_id}")
                    for attr_name, info in conn.parent_info['attributes'].items():
                        print(f"  Attribute '{attr_name}' inherited from parent {info['inherited_from']}: {info['value']}")
                else:
                    print(f"Connection {in_node} -> {out_node}: Inherited entirely from parent {conn.parent_info['inherited_from']}")
            else:
                print(f"Connection {in_node} -> {out_node}: No parent info available")
    '''
    def configure_crossover(self, genome1, genome2, config):
        """ Configure a new genome by crossover from two parent genomes. """
        if genome1.fitness > genome2.fitness:
            parent1, parent2 = genome1, genome2
        else:
            parent1, parent2 = genome2, genome1

        self.parents = [parent1.key, parent2.key]

        self.parent_fitness = parent1.fitness if parent1.fitness is not None else parent2.fitness

        # Inherit connection genes
        for key, cg1 in parent1.connections.items():
            cg2 = parent2.connections.get(key)
            if cg2 is None:
                # Excess or disjoint gene: copy from the fittest parent.
                self.connections[key] = cg1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.connections[key] = cg1.crossover(cg2, 1, 2)

        # Inherit node genes
        parent1_set = parent1.nodes
        parent2_set = parent2.nodes

        for key, ng1 in parent1_set.items():
            ng2 = parent2_set.get(key)
            assert key not in self.nodes
            if ng2 is None:
                # Extra gene: copy from the fittest parent
                self.nodes[key] = ng1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.nodes[key] = ng1.crossover(ng2, 1, 2, config.output_keys)

    def mutate(self, config):
        """ Mutates this genome. """

        if config.single_structural_mutation:
            div = max(1, (config.node_add_prob + config.node_delete_prob +
                          config.conn_add_prob + config.conn_delete_prob))
            r = random()
            if r < (config.node_add_prob / div):
                self.mutate_add_node(config)
            elif r < ((config.node_add_prob + config.node_delete_prob) / div):
                self.mutate_delete_node(config)
            elif r < ((config.node_add_prob + config.node_delete_prob +
                       config.conn_add_prob) / div):
                self.mutate_add_connection(config)
            elif r < ((config.node_add_prob + config.node_delete_prob +
                       config.conn_add_prob + config.conn_delete_prob) / div):
                self.mutate_delete_connection()
        else:
            if random() < config.node_add_prob:
                self.mutate_add_node(config)

            if random() < config.node_delete_prob:
                self.mutate_delete_node(config)

            if random() < config.conn_add_prob:
                self.mutate_add_connection(config)

            if random() < config.conn_delete_prob:
                self.mutate_delete_connection()

        # Mutate connection genes.
        for cg in self.connections.values():
            cg.mutate(config)

        # Mutate node genes (bias, response, etc.).
        for ng in self.nodes.values():
            ng.mutate(config)

    def mutate_add_node(self, config):
        if not self.connections:
            if config.check_structural_mutation_surer():
                self.mutate_add_connection(config)
            return

        # Choose a random enabled connection to split
        conn_to_split = choice(list(self.connections.values()))
        new_node_id = config.get_new_node_key(self.nodes)
        ng = self.create_node(config, new_node_id)
        self.nodes[new_node_id] = ng

        # Disable the original connection
        conn_to_split.enabled = False
        i, o = conn_to_split.key

        #print(f"Splitting connection from {i} to {o}")
        #print(f"New node ID: {ng}")

        if i in config.output_keys or o in config.input_keys:
            return  # Skip splitting this connection

        # Add new connections with correct innovation numbers
        # Note: get_innovation_number is called within add_connection
        # First, add connection from input node to new node
        cg_in = self.add_connection(
            config,
            input_key=i,
            output_key=new_node_id,
            weight_mu=config.weight_mu_init_mean,  # Typically set to 1.0 in NEAT
            weight_sigma=config.weight_sigma_init_mean,
            enabled=True
        )

        #if cg_in:
            #print(f"Successfully added connection from {i} to {ng}.")

        # Then, add connection from new node to output node
        cg_out = self.add_connection(
            config,
            input_key=new_node_id,
            output_key=o,
            weight_mu=conn_to_split.weight_mu,       # Preserve the original weight_mu
            weight_sigma=conn_to_split.weight_sigma, # Preserve the original weight_sigma
            enabled=True
        )

        #if cg_out:
            #print(f"Successfully added connection from {ng} to {o}.")

        # Optionally, log the mutation
        self.mutation_history.append({
            'type': 'add_node',
            'genome_id': self.key,  # Use self.key as the genome identifier
            'split_connection': conn_to_split.key,
            'new_node': ng,
            'new_connections': [
                {
                    'input': i,
                    'output': new_node_id,
                    'weight_mu': cg_in.weight_mu,
                    'weight_sigma': cg_in.weight_sigma
                },
                {
                    'input': new_node_id,
                    'output': o,
                    'weight_mu': cg_out.weight_mu,
                    'weight_sigma': cg_out.weight_sigma
                }
            ]
        })


    def add_connection(self, config, input_key, output_key, weight_mu, weight_sigma, enabled):
        # TODO: Add further validation of this connection addition?
        assert isinstance(input_key, int)
        assert isinstance(output_key, int)
        assert output_key >= 0
        assert isinstance(enabled, bool)
        key = (input_key, output_key)
        # Get or assign an innovation number

        connection = config.connection_gene_type(key)
        connection.init_attributes(config)
        connection.weight_mu = weight_mu
        connection.weight_sigma = weight_sigma
        connection.enabled = enabled
        self.connections[key] = connection
        #print(f"Added connection {key} to self.connections:")
        return connection

    def mutate_add_connection(self, config):
        """
        Attempt to add a new connection, the only restriction being that the output
        node cannot be one of the network input pins.
        """
        possible_outputs = [node_id for node_id in self.nodes if node_id not in config.input_keys]
        if not possible_outputs:
            return  # No valid output nodes available

        out_node = choice(possible_outputs)

        possible_inputs = [node_id for node_id in self.nodes if node_id not in config.output_keys] + config.input_keys
        if not possible_inputs:
            return  # No valid input nodes available
        in_node = choice(possible_inputs)

        if out_node in config.input_keys:
            return

        # Don't duplicate connections.
        key = (in_node, out_node)
        if key in self.connections:
            # TODO: Should this be using mutation to/from rates? Hairy to configure...
            if config.check_structural_mutation_surer():
                self.connections[key].enabled = True
            return

        # Don't allow connections between two output nodes
        if in_node in config.output_keys and out_node in config.output_keys:
            return

        # No need to check for connections between input nodes:
        # they cannot be the output end of a connection (see above).

        # For feed-forward networks, avoid creating cycles.
        if config.feed_forward and creates_cycle(list(self.connections.keys()), key):
            return

        cg = self.create_connection(config, in_node, out_node, enabled=True)
        self.connections[cg.key] = cg

        # **Append to mutation history**
        self.mutation_history.append({
            'type': 'add_connection',
            'genome_id': self.key,  # Assuming self.key uniquely identifies each genome
            'connection': (in_node, out_node),
            'weight_mu': cg.weight_mu,
            'weight_sigma': cg.weight_sigma
        })


        #print(f"Genome {self.key}: Added connection from {in_node} to {out_node} with weight mu: {cg.weight_mu}, sigma: {cg.weight_sigma}")


    def mutate_delete_node(self, config):
        # Do nothing if there are no non-output nodes.
        available_nodes = [k for k in self.nodes if k not in config.output_keys and k not in config.input_keys]
        #print("config.input_keys: ", config.input_keys)
        if not available_nodes:
            return -1

        del_key = choice(available_nodes)

        connections_to_delete = set()
        for k, v in self.connections.items():
            if del_key in v.key:
                connections_to_delete.add(v.key)

        for key in connections_to_delete:
            del self.connections[key]

        del self.nodes[del_key]

        # **Append to mutation history**
        self.mutation_history.append({
            'type': 'delete_node',
            'genome_id': self.key,
            'deleted_node': del_key,
            'deleted_connections': list(connections_to_delete)
        })

        # Track the mutation in the mutation history
        #print(f"Genome {self.key}: Deleted node {del_key} and connections {list(connections_to_delete)}")

        return del_key

    def mutate_delete_connection(self):
        if self.connections:
            key = choice(list(self.connections.keys()))

            self.mutation_history.append({
                'type': 'delete_connection',
                'genome_id': self.key,
                'deleted_connection': key,
                'weight_mu': self.connections[key].weight_mu,
                'weight_sigma': self.connections[key].weight_sigma
            })

            #print(f"Genome {self.key}: Deleted connection {key} with weight mu: {self.connections[key].weight_mu}, sigma: {self.connections[key].weight_sigma}")

            del self.connections[key]

    def distance(self, other, config):
        """
        Returns the genetic distance between this genome and the other. This distance value
        is used to compute genome compatibility for speciation.
        """

        # Compute node gene distance component.
        node_distance = 0.0
        if self.nodes or other.nodes:
            disjoint_nodes = 0
            for k2 in other.nodes:
                if k2 not in self.nodes:
                    disjoint_nodes += 1

            for k1, n1 in self.nodes.items():
                n2 = other.nodes.get(k1)
                if n2 is None:
                    disjoint_nodes += 1
                else:
                    # Homologous genes compute their own distance value.
                    node_distance += n1.distance(n2, config)

            max_nodes = max(len(self.nodes), len(other.nodes))
            node_distance = (node_distance +
                             (config.compatibility_disjoint_coefficient *
                              disjoint_nodes)) / max_nodes

        # Compute connection gene differences.
        connection_distance = 0.0
        if self.connections or other.connections:
            disjoint_connections = 0
            for k2 in other.connections:
                if k2 not in self.connections:
                    disjoint_connections += 1

            for k1, c1 in self.connections.items():
                c2 = other.connections.get(k1)
                if c2 is None:
                    disjoint_connections += 1
                else:
                    # Homologous genes compute their own distance value.
                    connection_distance += c1.distance(c2, config)

            max_conn = max(len(self.connections), len(other.connections))
            connection_distance = (connection_distance +
                                   (config.compatibility_disjoint_coefficient *
                                    disjoint_connections)) / max_conn

        distance = node_distance + connection_distance
        return distance

    def size(self):
        """
        Returns genome 'complexity', taken to be
        (number of nodes, number of enabled connections)
        """
        num_enabled_connections = sum([1 for cg in self.connections.values() if cg.enabled])
        return len(self.nodes), num_enabled_connections

    def __str__(self):
        s = f"Key: {self.key}\nFitness: {self.fitness}\nNodes:"
        for k, ng in self.nodes.items():
            s += f"\n\t{k} {ng!s}"
        s += "\nConnections:"
        connections = list(self.connections.values())
        connections.sort()
        for c in connections:
            s += "\n\t" + str(c)
        return s

    @staticmethod
    def create_node(config, node_id):
        node = config.node_gene_type(node_id)
        node.init_attributes(config)

        # Check for Bayesian attributes
        assert hasattr(node, 'bias_mu'), "Node does not have bias_mu attribute"
        assert hasattr(node, 'bias_sigma'), "Node does not have bias_sigma attribute"
    
        node.initialized = True  # Mark the node as initialized
        #print(f"Created node {node_id} with bias_mu={node.bias_mu}, bias_sigma={node.bias_sigma}")
        return node

    @staticmethod
    def create_connection(config, input_id, output_id, weight_mu=None, weight_sigma=None, enabled=True):
        key = (input_id, output_id)

        # Get or assign an innovation number

        # Create the connection gene
        connection = config.connection_gene_type((input_id, output_id))

        # Initialize the connection's attributes (weight_mu, weight_sigma, etc.)
        connection.init_attributes(config)

        # Check for Bayesian attributes
        assert hasattr(connection, 'weight_mu'), "Connection does not have weight_mu attribute"
        assert hasattr(connection, 'weight_sigma'), "Connection does not have weight_sigma attribute"


        # Mark the connection as initialized to avoid re-initialization
        connection.initialized = True

        # Set `weight_mu` and `weight_sigma` if provided
        if weight_mu is not None:
            connection.weight_mu = weight_mu
        if weight_sigma is not None:
            connection.weight_sigma = weight_sigma
        # Otherwise, `weight_mu` and `weight_sigma` remain as initialized
        # Set enabled status
        connection.enabled = enabled
        return connection

    def connect_fs_neat_nohidden(self, config):
        """
        Randomly connect one input to all output nodes
        (FS-NEAT without connections to hidden, if any).
        Originally connect_fs_neat.
        """
        input_id = choice(config.input_keys)
        for output_id in config.output_keys:
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def connect_fs_neat_hidden(self, config):
        """
        Randomly connect one input to all hidden and output nodes
        (FS-NEAT with connections to hidden, if any).
        """
        input_id = choice(config.input_keys)
        others = [i for i in self.nodes if i not in config.input_keys]
        for output_id in others:
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def compute_full_connections(self, config, direct):
        """
        Compute connections for a fully-connected feed-forward genome--each
        input connected to all hidden nodes
        (and output nodes if ``direct`` is set or there are no hidden nodes),
        each hidden node connected to all output nodes.
        (Recurrent genomes will also include node self-connections.)
        """
        hidden = [i for i in self.nodes if i not in config.output_keys]
        output = [i for i in self.nodes if i in config.output_keys]
        connections = []
        if hidden:
            for input_id in config.input_keys:
                for h in hidden:
                    connections.append((input_id, h))
            for h in hidden:
                for output_id in output:
                    connections.append((h, output_id))
        if direct or (not hidden):
            for input_id in config.input_keys:
                for output_id in output:
                    connections.append((input_id, output_id))

        # For recurrent genomes, include node self-connections.
        if not config.feed_forward:
            for i in self.nodes:
                connections.append((i, i))

        return connections

    def connect_full_nodirect(self, config):
        """
        Create a fully-connected genome
        (except without direct input-output unless no hidden nodes).
        """
        for input_id, output_id in self.compute_full_connections(config, False):
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def connect_full_direct(self, config):
        """ Create a fully-connected genome, including direct input-output connections. """
        for input_id, output_id in self.compute_full_connections(config, True):
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def connect_partial_nodirect(self, config):
        """
        Create a partially-connected genome,
        with (unless no hidden nodes) no direct input-output connections.
        """
        assert 0 <= config.connection_fraction <= 1
        all_connections = self.compute_full_connections(config, False)
        shuffle(all_connections)
        num_to_add = int(round(len(all_connections) * config.connection_fraction))
        for input_id, output_id in all_connections[:num_to_add]:
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def connect_partial_direct(self, config):
        """
        Create a partially-connected genome,
        including (possibly) direct input-output connections.
        """
        assert 0 <= config.connection_fraction <= 1
        all_connections = self.compute_full_connections(config, True)
        shuffle(all_connections)
        num_to_add = int(round(len(all_connections) * config.connection_fraction))
        for input_id, output_id in all_connections[:num_to_add]:
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def get_pruned_copy(self, genome_config):
        print("PRUNE")
        used_node_genes, used_connection_genes = get_pruned_genes(self.nodes, self.connections,
                                                                  genome_config.input_keys, genome_config.output_keys)
        new_genome = DefaultGenome(None)
        new_genome.nodes = used_node_genes
        new_genome.connections = used_connection_genes
        return new_genome

    def update_parameters(self, optimized_params):
        """
        Updates the genome's node and connection genes with the optimized parameters from SVI.
        optimized_params is expected to be a dictionary containing 'nodes' and 'connections',
        where each maps to the appropriate mu and sigma values.
        """
        # Update node genes
        for node_key, node_gene in self.nodes.items():
            if node_key in optimized_params['nodes']:
                # Update the node with new mu and sigma from SVI optimization
                node_gene.update_parameters(optimized_params['nodes'][node_key])
                node_gene.initialized = True  # Mark as initialized after updating
            else:
                # If the node does not have optimized params (e.g., newly evolved), initialize it
                node_gene.init_attributes(self.config.genome_config)
                node_gene.initialized = True  # Mark as initialized after updating

        # Update connection genes with optimized parameters
        for conn_key, conn_gene in self.connections.items():
            if conn_key in optimized_params['connections']:
                # Update the connection with new mu and sigma from SVI optimization
                conn_gene.update_parameters(optimized_params['connections'][conn_key])
                conn_gene.initialized = True  # Mark as initialized after updating
            else:
                # If the connection does not have optimized params (e.g., newly evolved), initialize it
                conn_gene.init_attributes(self.config.genome_config)
                conn_gene.initialized = True  # Mark as initialized after updating

def get_pruned_genes(node_genes, connection_genes, input_keys, output_keys):
    print("PRUNE")
    used_nodes = required_for_output(input_keys, output_keys, connection_genes)
    used_pins = used_nodes.union(input_keys)

    # Copy used nodes into a new genome.
    used_node_genes = {}
    for n in used_nodes:
        used_node_genes[n] = copy.deepcopy(node_genes[n])

    # Copy enabled and used connections into the new genome.
    used_connection_genes = {}
    for key, cg in connection_genes.items():
        in_node_id, out_node_id = key
        if cg.enabled and in_node_id in used_pins and out_node_id in used_pins:
            used_connection_genes[key] = copy.deepcopy(cg)

    return used_node_genes, used_connection_genes

