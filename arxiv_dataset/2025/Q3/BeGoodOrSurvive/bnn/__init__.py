# Import the BayesianNN class to make it directly accessible via bnn.BayesianNN
from .bayesnn import BayesianNN

# If there are utility functions in utils.py, you can import them as well
from .bnn_utils import update_bnn_history, get_activation_function, get_aggregation_function  # Replace with actual function names

from .attention import compute_attention

