import torch
import json

def adjust_rates_proportional(config, neat_iteration, total_iterations, initial_rates, final_rates):
    """
    Adjust rates proportionally based on the initial values, ensuring no compounding errors.
    """
    print(initial_rates)
    for rate_name in initial_rates:
        print(rate_name, type(rate_name))
        initial_rate = initial_rates[rate_name]
        print(initial_rate)
        final_rate = final_rates[rate_name]
        print(final_rate)
        delta_rate = (initial_rate - final_rate) / total_iterations
        print(delta_rate)
        new_rate = max(initial_rate - neat_iteration * delta_rate, final_rate)

        # Update the configuration (directly modifies the internal state)
        setattr(config.genome_config, rate_name, new_rate)
        print(f"Adjusted {rate_name}: {new_rate:.4f}")

    return config

def save_evolution_results(results, tradeoffs, neat_iteration, file_path_template="124_prod_evolution_results_iter_{gen}.json"):
    file_path=f"recycle_prod_evolution_results_iter_{neat_iteration}.json"
    # Convert objects to JSON-compatible formats
    def convert(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        else:
            try:
                return str(obj)
            except:
                print("messed iup in save evolutoon results")
                raise ValueError

    # Combine results and tradeoffs
    complete_results = {
        "results": results,
        "tradeoffs": tradeoffs
    }

    results_serializable = convert(complete_results)

    print("SAVE EVOLUTION RESULTS FILE PATH: ", file_path)

    # Save to JSON
    with open(file_path, "w") as file:
        json.dump(results_serializable, file, indent=4)

