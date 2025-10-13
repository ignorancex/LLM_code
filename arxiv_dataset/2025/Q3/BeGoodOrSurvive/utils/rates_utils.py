def get_initial_rates():
    """
    Returns a dictionary containing the initial rates for parameters 
    with high starting values.
    """
    return {
        # Bias mutation and replacement
        "bias_mu_mutate_rate": 0.9,
        "bias_mu_mutate_power": 1.0,
        "bias_mu_replace_rate": 0.5,
        "bias_sigma_mutate_rate": 0.9,
        "bias_sigma_mutate_power": 0.9,
        "bias_sigma_replace_rate": 0.5,

        # Weight mutation and replacement
        "weight_mu_mutate_rate": 0.9,
        "weight_mu_mutate_power": 1.0,
        "weight_mu_replace_rate": 0.5,
        "weight_sigma_mutate_rate": 0.9,
        "weight_sigma_mutate_power": 1.0,
        "weight_sigma_replace_rate": 0.5,

        # Response mutation and replacement
        "response_mu_mutate_rate": 0.9,
        "response_mu_mutate_power": 0.9,
        "response_mu_replace_rate": 0.5,
        "response_sigma_mutate_rate": 0.9,
        "response_sigma_mutate_power": 0.5,
        "response_sigma_replace_rate": 0.5,

        # Node mutation
        "node_add_prob": 0.9,
        "node_delete_prob": 0.7,

        # Connection mutation
        "conn_add_prob": 0.9,
        "conn_delete_prob": 0.9,

        "compatibility_threshold": 13.0,
        "max_stagnation": 10
    }

def get_final_rates():
    """
    Returns a dictionary containing the final rates for parameters
    with low ending values for gradual decay.
    """
    return {
        # Bias mutation and replacement
        "bias_mu_mutate_rate": 0.9,
        "bias_mu_mutate_power": 1.0,
        "bias_mu_replace_rate": 0.5,
        "bias_sigma_mutate_rate": 0.9,
        "bias_sigma_mutate_power": 0.9,
        "bias_sigma_replace_rate": 0.5,

        # Weight mutation and replacement
        "weight_mu_mutate_rate": 0.9,
        "weight_mu_mutate_power": 1.0,
        "weight_mu_replace_rate": 0.5,
        "weight_sigma_mutate_rate": 0.9,
        "weight_sigma_mutate_power": 1.0,
        "weight_sigma_replace_rate": 0.5,

        # Response mutation and replacement
        "response_mu_mutate_rate": 0.9,
        "response_mu_mutate_power": 0.9,
        "response_mu_replace_rate": 0.5,
        "response_sigma_mutate_rate": 0.9,
        "response_sigma_mutate_power": 0.5,
        "response_sigma_replace_rate": 0.5,

        # Node mutation
        "node_add_prob": 0.9,
        "node_delete_prob": 0.7,

        # Connection mutation
        "conn_add_prob": 0.9,
        "conn_delete_prob": 0.9,

        "compatibility_threshold": 13.0,
        "max_stagnation": 10
    }

'''
def get_final_rates():
    """
    Returns a dictionary containing the final rates for parameters 
    with low ending values for gradual decay.
    """
    return {
        # Bias mutation and replacement
        "bias_mu_mutate_rate": 0.15,
        "bias_mu_mutate_power": 0.1,
        "bias_mu_replace_rate": 0.1,
        "bias_sigma_mutate_rate": 0.15,
        "bias_sigma_mutate_power": 0.1,
        "bias_sigma_replace_rate": 0.1,

        # Weight mutation and replacement
        "weight_mu_mutate_rate": 0.15,
        "weight_mu_mutate_power": 0.1,
        "weight_mu_replace_rate": 0.1,
        "weight_sigma_mutate_rate": 0.15,
        "weight_sigma_mutate_power": 0.1,
        "weight_sigma_replace_rate": 0.1,

        # Response mutation and replacement
        "response_mu_mutate_rate": 0.15,
        "response_mu_mutate_power": 0.1,
        "response_mu_replace_rate": 0.1,
        "response_sigma_mutate_rate": 0.1,
        "response_sigma_mutate_power": 0.1,
        "response_sigma_replace_rate": 0.1,

        # Node mutation
        "node_add_prob": 0.2,
        "node_delete_prob": 0.2,

        # Connection mutation
        "conn_add_prob": 0.2,
        "conn_delete_prob": 0.2,

        "compatibility_threshold": 4,
        "max_stagnation": 6,
    }
'''

def print_config_values(config):
    """
    Prints all relevant configuration values from the config object.
    """
    final_rates = get_final_rates()  # Fetch final rates for reference
    relevant_keys = final_rates.keys()  # Use the keys from final_rates as a reference for relevant parameters
    
    print("Current Config Values:")
    for key in relevant_keys:
        value = getattr(config.genome_config, key, "Key not found")  # Get the value from the config
        print(f"{key}: {value}")

    # Additional parameters that are not mutation rates
    print("\n=== Additional Config Parameters ===")
    
    additional_params = {
        "compatibility_threshold": getattr(config.species_set_config, "compatibility_threshold", "Key not found"),
        "max_stagnation": getattr(config.stagnation_config, "max_stagnation", "Key not found"),
    }

    for param, value in additional_params.items():
        print(f"{param}: {value}")
