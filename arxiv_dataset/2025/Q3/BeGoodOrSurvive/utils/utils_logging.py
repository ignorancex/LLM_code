import json
import torch

# In logging_utils.py
def save_experiment_results(result, loss, survival, ethics, ethical_ground_truths, survival_rate, filename="124_prod_experiment_results.json"):
    data_to_save = {
        "result": result,
        "loss_per_generation": loss,
        "survival_history": survival,
        "svi_decision_ethics": ethics,
        "ethical_ground_truths": ethical_ground_truths,
        "survival_rate": survival_rate,
        "progress": {
            "average_loss_per_gen": [sum(l) / len(l) if l else 0 for l in loss],
            "average_ethical_score_per_gen": [sum(e) / len(e) if len(e) > 0 else 0 for e in ethics],
            "survival_counts_per_gen": list(survival.values()),
        }
    }
    with open(filename, "w") as file:
        json.dump(data_to_save, file, indent=4)
    print(f"Results saved to '{filename}'")

def make_population_tradeoffs_serializable(population_tradeoffs):
    """
    Converts objects in population tradeoffs to JSON-compatible formats.
    """
    def convert(obj):
        # Handle known types explicitly
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):  # Basic JSON types
            return obj
        else:
            # Fallback for unexpected types
            return str(obj)  # Use str for anything not explicitly handled

    # Recursively apply conversion to all entries in the tradeoffs
    return [convert(tradeoff) for tradeoff in population_tradeoffs]
