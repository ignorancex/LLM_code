from sparselearning.pruners import *
from sparselearning.growers import *
from sparselearning.redistributions import *
from sparselearning.sparsity_initializers import *
from sparselearning.weight_preprocesser import *

prune_funcs = {
    'magnitude': magnitude_prune,
    'SET': magnitude_and_negativity_prune,
    'global_magnitude': global_magnitude_prune
}

growth_funcs = {
    'random': random_growth,
    'momentum': momentum_growth,
    'momentum_neuron': momentum_neuron_growth,
    'gradient': gradient_growth_fixed,
    'global_momentum_growth': global_momentum_growth
}

redistribution_funcs = {
    'momentum': momentum_redistribution,
    'nonzero': nonzero_redistribution,
    'magnitude': magnitude_redistribution,
    'none': no_redistribution
}

sparsity_inits = {

    "global_magnitude": global_magnitude_initializer,
    "snip": snip_initializer,
    "grasp": grasp_initializer,
    "snip_direct": direct_snip_initializer,
    "grasp_direct": direct_grasp_initializer,
    "GraSP": grasp_initializer,
    "synflow": synflow_initializer,
    "uniform_plus": uniform_plus_initializer,
    "uniform": uniform_initializer,
    "er": ERK_initializer,
    "erk": ERK_initializer,
    "from_weight": from_weight_initializer,
    "from_weights": from_weight_initializer,
    "uniform_AI": uniform_initializer,
    "erk_AI": ERK_initializer,
    "snip_direct_AI": direct_snip_initializer,
    "grasp_direct_AI": direct_grasp_initializer,
    "uniform_EI": uniform_initializer,
    "erk_EI": ERK_initializer,
    "snip_direct_EI": direct_snip_initializer,
    "grasp_direct_EI": direct_grasp_initializer,
    "synflow_direct": direct_synflow_initializer,
    "synflow_direct_EI": direct_synflow_initializer,
    "synflow_direct_AI": direct_synflow_initializer,
    "uniform_SAO": uniform_initializer,
    "erk_EIS": ERK_initializer,
    "uni_skip_EI": uniform_skip_first_last_initializer,
    "uni_skip_EIS": uniform_skip_first_last_initializer
}
