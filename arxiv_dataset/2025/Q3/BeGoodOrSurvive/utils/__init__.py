# utils/__init__.py

# Import specific functions from text_utils.py
from .text_utils import normalize_string, trim_response, extract_choices_and_intro

# Import specific utilities from logging.py
from .utils_logging import save_experiment_results  # Example: Replace with actual function/class names in logging.py

# Import plotting utilities
from .plotting import plot_loss_and_survival, plot_survival_and_ethics, plot_loss_and_ethics

# Import text generation utilities
from .text_generation import generate_text  # Example: Replace with actual text generation function/class

from .rates_utils import get_initial_rates, get_final_rates

from .checkpoint import save_checkpoint, load_checkpoint

# If desired, define what gets imported when using `from utils import *`
__all__ = [
    "normalize_string",
    "trim_response",
    "extract_choices_and_intro",
    "setup_logger",
    "plot_loss",
    "plot_accuracy",
    "generate_text",
    "get_initial_rates",
    "get_final_rates",
]
