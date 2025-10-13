"""
Multi-Expert Chain for Audio Tasks (MECAT)
"""

__version__ = "0.1.0"

# Use lazy imports to avoid early import conflicts
def __getattr__(name):
    """Lazy import mechanism to avoid early imports"""
    if name == "evaluate":
        from .evaluate import evaluate
        return evaluate
    elif name == "load_data_with_config":
        from .evaluate import load_data_with_config
        return load_data_with_config
    elif name == "load_prediction_file":
        from .evaluate import load_prediction_file
        return load_prediction_file
    elif name == "results_to_dataframe":
        from .evaluate import results_to_dataframe
        return results_to_dataframe
    elif name == "BaseScorer":
        from .scorer import BaseScorer
        return BaseScorer
    elif name == "DATE":
        from .DATE import DATE
        return DATE
    elif name == "DATEEvaluator":
        from .DATE import DATEEvaluator
        return DATEEvaluator
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "evaluate",
    "load_data_with_config",
    "load_prediction_file",
    "BaseScorer",
    "results_to_dataframe",
    "DATE",
    "DATEEvaluator",
]
