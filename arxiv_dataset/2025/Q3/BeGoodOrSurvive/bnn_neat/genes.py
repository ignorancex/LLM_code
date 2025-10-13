"""Handles node and connection genes."""
import warnings
import random

from bnn_neat.attributes import FloatAttribute, BoolAttribute, StringAttribute


# TODO: There is probably a lot of room for simplification of these classes using metaprogramming.
# TODO: Evaluate using __slots__ for performance/memory usage improvement.


class BaseGene(object):
    """
    Handles functions shared by multiple types of genes (both node and connection),
    including crossover and calling mutation methods.
    """

    def __init__(self, key, parent_info=None):
        self.key = key
        self.parent_info = parent_info
        self.initialized = False

    def __str__(self):
        attrib = ['key'] + [a.name for a in self._gene_attributes]
        attrib = [f'{a}={getattr(self, a)}' for a in attrib]
        return f'{self.__class__.__name__}({", ".join(attrib)})'

    def __lt__(self, other):
        assert isinstance(self.key, type(other.key)), f"Cannot compare keys {self.key!r} and {other.key!r}"
        return self.key < other.key

    @classmethod
    def parse_config(cls, config, param_dict):
        pass

    @classmethod
    def get_config_params(cls):
        params = []
        if not hasattr(cls, '_gene_attributes'):
            setattr(cls, '_gene_attributes', getattr(cls, '__gene_attributes__'))
            warnings.warn(
                f"Class '{cls.__name__!s}' {cls!r} needs '_gene_attributes' not '__gene_attributes__'",
                DeprecationWarning)
        for a in cls._gene_attributes:
            params += a.get_config_params()
        return params

    @classmethod
    def validate_attributes(cls, config):
        for a in cls._gene_attributes:
            a.validate(config)

    def init_attributes(self, config):
        if hasattr(self, 'initialized') and self.initialized:
            # Skip re-initialization to preserve existing parameters
            return
        for a in self._gene_attributes:
            if a.name in ('weight_mu', 'bias_mu', 'response_mu'):
                # Initialize the mean using the corresponding config parameters
                mean = getattr(config, f'{a.name}_init_mean')
                stdev = getattr(config, f'{a.name}_init_stdev')
                setattr(self, a.name, random.gauss(mean, stdev))
            elif a.name in ('weight_sigma', 'bias_sigma', 'response_sigma'):
                # Initialize the sigma (std dev), ensuring it stays positive
                init_mean = getattr(config, f'{a.name}_init_mean')
                init_stdev = getattr(config, f'{a.name}_init_stdev')
                min_value = getattr(config, f'{a.name}_min_value')
                sigma_value = max(min_value, random.gauss(init_mean, init_stdev))
                setattr(self, a.name, sigma_value)
            else:
                # For other attributes, fall back to the default initialization
                setattr(self, a.name, a.init_value(config))

        # Mark the gene as initialized
        self.initialized = True



# TODO: Should these be in the nn module?  iznn and ctrnn can have additional attributes.


class DefaultNodeGene(BaseGene):
    _gene_attributes = [FloatAttribute('bias_mu'),
                        FloatAttribute('bias_sigma'),
                        FloatAttribute('response_mu'),
                        FloatAttribute('response_sigma'),
                        StringAttribute('activation', options='relu sigmoid tanh'),
                        StringAttribute('aggregation', options='sum product max min mean')]

    def __init__(self, key):
        assert isinstance(key, int), f"DefaultNodeGene key must be an int, not {key!r}"
        BaseGene.__init__(self, key)

    def distance(self, other, config):
        # Distance between bias means and standard deviations
        d_bias_mu = abs(self.bias_mu - other.bias_mu)
        d_bias_sigma = abs(self.bias_sigma - other.bias_sigma)

        # Distance between response means and standard deviations
        d_response_mu = abs(self.response_mu - other.response_mu)
        d_response_sigma = abs(self.response_sigma - other.response_sigma)

        # Total distance
        d = d_bias_mu + d_bias_sigma + d_response_mu + d_response_sigma

        # Check for differences in activation and aggregation functions
        if self.activation != other.activation:
            d += 1.0
        if self.aggregation != other.aggregation:
            d += 1.0

        return d * config.compatibility_weight_coefficient

    def mutate(self, config):
        # Mutate bias_mu
        if random.random() < config.bias_mu_replace_rate:
            self.bias_mu = random.uniform(config.bias_mu_min_value, config.bias_mu_max_value)
        elif random.random() < config.bias_mu_mutate_rate:
            self.bias_mu += random.gauss(0.0, config.bias_mu_mutate_power)
            self.bias_mu = max(min(self.bias_mu, config.bias_mu_max_value), config.bias_mu_min_value)

        # Mutate bias_sigma
        if random.random() < config.bias_sigma_replace_rate:
            self.bias_sigma = random.uniform(config.bias_sigma_min_value, config.bias_sigma_max_value)
        elif random.random() < config.bias_sigma_mutate_rate:
            self.bias_sigma += random.gauss(0.0, config.bias_sigma_mutate_power)
            self.bias_sigma = max(min(self.bias_sigma, config.bias_sigma_max_value), config.bias_sigma_min_value)

        # Mutate response_mu
        if random.random() < config.response_mu_replace_rate:
            self.response_mu = random.uniform(config.response_mu_min_value, config.response_mu_max_value)
        elif random.random() < config.response_mu_mutate_rate:
            self.response_mu += random.gauss(0.0, config.response_mu_mutate_power)
            self.response_mu = max(min(self.response_mu, config.response_mu_max_value), config.response_mu_min_value)

        # Mutate response_sigma
        if random.random() < config.response_sigma_replace_rate:
            self.response_sigma = random.uniform(config.response_sigma_min_value, config.response_sigma_max_value)
        elif random.random() < config.response_sigma_mutate_rate:
            self.response_sigma += random.gauss(0.0, config.response_sigma_mutate_power)
            self.response_sigma = max(min(self.response_sigma, config.response_sigma_max_value), config.response_sigma_min_value)

        # Mutate activation function
        if random.random() < config.activation_mutate_rate:
            self.activation = random.choice(config.activation_options)

        # Mutate aggregation function
        if random.random() < config.aggregation_mutate_rate:
            self.aggregation = random.choice(config.aggregation_options)

    def update_parameters(self, new_mu_sigma):
        """
        Updates the node's parameters with new mu and sigma values for bias and response.
        new_mu_sigma is expected to be a dictionary with keys 'bias_mu', 'bias_sigma',
        'response_mu', and 'response_sigma'.
        """
        self.bias_mu = new_mu_sigma.get('bias_mu', self.bias_mu)
        self.bias_sigma = new_mu_sigma.get('bias_sigma', self.bias_sigma)
        self.response_mu = new_mu_sigma.get('response_mu', self.response_mu)
        self.response_sigma = new_mu_sigma.get('response_sigma', self.response_sigma)

    def crossover(self, gene2, parent1_id, parent2_id, output_node_ids):
            """Creates a new node gene by randomly inheriting attributes from the parents."""
            assert self.key == gene2.key

            # Create a new node gene
            new_gene = self.__class__(self.key)

            # Initialize parent_info to store attribute-level inheritance
            new_gene.parent_info = {
                'parents': [parent1_id, parent2_id],
                'attributes': {}
            }

            for a in self._gene_attributes:
                value1 = getattr(self, a.name)
                value2 = getattr(gene2, a.name)

                if random.random() > 0.5:
                    # Inherit from self (parent1)
                    setattr(new_gene, a.name, value1)
                    new_gene.parent_info['attributes'][a.name] = {
                        'inherited_from': parent1_id,
                        'value': value1
                    }
                else:
                    # Inherit from gene2 (parent2)
                    setattr(new_gene, a.name, value2)
                    new_gene.parent_info['attributes'][a.name] = {
                        'inherited_from': parent2_id,
                        'value': value2
                    }

            if self.key in output_node_ids:
                # Set activation function to 'sigmoid' or the desired fixed function
                new_gene.activation = 'sigmoid'

            return new_gene

    def copy(self):
        new_gene = self.__class__(self.key)
        for a in self._gene_attributes:
            setattr(new_gene, a.name, getattr(self, a.name))

        return new_gene

# TODO: Do an ablation study to determine whether the enabled setting is
# important--presumably mutations that set the weight to near zero could
# provide a similar effect depending on the weight range, mutation rate,
# and aggregation function. (Most obviously, a near-zero weight for the
# product aggregation function is rather more important than one giving
# an output of 1 from the connection, for instance!)
class DefaultConnectionGene(BaseGene):
    _gene_attributes = [FloatAttribute('weight_mu'),
                        FloatAttribute('weight_sigma'),
                        BoolAttribute('enabled')]

    def __init__(self, key):
        assert isinstance(key, tuple), f"DefaultConnectionGene key must be a tuple, not {key!r}"
        BaseGene.__init__(self, key)

        self.enabled = True

    def distance(self, other, config):
        # Distance between weight means and standard deviations
        d_weight_mu = abs(self.weight_mu - other.weight_mu)
        d_weight_sigma = abs(self.weight_sigma - other.weight_sigma)

        # Total distance
        d = d_weight_mu + d_weight_sigma

        if self.enabled != other.enabled:
            d += 1.0

        return d * config.compatibility_weight_coefficient


    def mutate(self, config):
        self.enabled = bool(self.enabled)

        # Mutate weight_mu
        if random.random() < config.weight_mu_replace_rate:
            self.weight_mu = random.uniform(config.weight_mu_min_value, config.weight_mu_max_value)
        elif random.random() < config.weight_mu_mutate_rate:
            self.weight_mu += random.gauss(0.0, config.weight_mu_mutate_power)
            self.weight_mu = max(min(self.weight_mu, config.weight_mu_max_value), config.weight_mu_min_value)


        # Mutate weight_sigma
        if random.random() < config.weight_sigma_replace_rate:
            self.weight_sigma = random.uniform(config.weight_sigma_min_value, config.weight_sigma_max_value)
        elif random.random() < config.weight_sigma_mutate_rate:
            self.weight_sigma += random.gauss(0.0, config.weight_sigma_mutate_power)
            self.weight_sigma = max(min(self.weight_sigma, config.weight_sigma_max_value), config.weight_sigma_min_value)

        # Mutate enabled flag
        if random.random() < config.enabled_mutate_rate:
            if self.enabled:
                self.enabled = False
            else:
                self.enabled = True

    def update_parameters(self, new_mu_sigma):
            """
            Updates the connection's parameters with new mu and sigma values for weights.
            new_mu_sigma is expected to be a dictionary with keys 'weight_mu' and 'weight_sigma'.
            """
            self.weight_mu = new_mu_sigma.get('weight_mu', self.weight_mu)
            self.weight_sigma = new_mu_sigma.get('weight_sigma', self.weight_sigma)

    def crossover(self, gene2, parent1_id, parent2_id):
        """Creates a new connection gene by randomly inheriting attributes from the parents."""
        assert self.key == gene2.key

        new_gene = self.__class__(self.key)

        # Initialize parent_info to store attribute-level inheritance
        new_gene.parent_info = {
            'parents': [parent1_id, parent2_id],
            'attributes': {}
        }

        for a in self._gene_attributes:
            value1 = getattr(self, a.name)
            value2 = getattr(gene2, a.name)

            if random.random() > 0.5:
                # Inherit from self (parent1)
                setattr(new_gene, a.name, value1)
                new_gene.parent_info['attributes'][a.name] = {
                    'inherited_from': parent1_id,
                    'value': value1
                }
            else:
                # Inherit from gene2 (parent2)
                setattr(new_gene, a.name, value2)
                new_gene.parent_info['attributes'][a.name] = {
                    'inherited_from': parent2_id,
                    'value': value2
                }

        return new_gene

    def copy(self):
        new_gene = self.__class__(self.key)
        for a in self._gene_attributes:
            setattr(new_gene, a.name, getattr(self, a.name))

        return new_gene
