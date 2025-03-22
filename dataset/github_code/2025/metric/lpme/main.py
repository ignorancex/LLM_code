import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

class DataDistribution:
    def __init__(self, N: int):
        """
        Initializes the data distribution with a specified number of samples.
        
        Args:
        - N (int): Number of data points.
        """
        self.weights = torch.tensor([-0.3356, -1.4104, 0.3144, -0.5591, 1.0426, 0.6036, -0.7549, -1.1909, 1.4779, -0.7513])
        self.D = len(self.weights)

        gen = torch.Generator().manual_seed(42)
        self.data = torch.randn(N, self.D, generator=gen)
        self.probs = torch.sigmoid(self.data @ self.weights)
    
def classifier_metrics(data_dist, threshold, upper=True):
    """
    Computes the True Positive and True Negative rates based on a classifier threshold.
    
    Args:
    - data_dist (DataDistribution): The data distribution instance.
    - threshold (float): Threshold value for classification.
    - upper (bool): If True, classifies as positive if above threshold; else, if below.
    
    Returns:
    - tuple (float, float): True Positive Rate (TP) and True Negative Rate (TN) in that order.
    """
    # YOUR CODE HERE (~3-5 lines)
    probs = data_dist.probs
    is_positive = ((probs > threshold) ^ (not upper)).float()
    tp = (probs *  is_positive).mean()
    tn = ((1 - probs) * (1 - is_positive)).mean()
    return tp.item(), tn.item()
    # END OF YOUR CODE

def sweep_classifiers(data_dist: DataDistribution):
    """
    Sweeps through classifier thresholds and calculates True Positive and True Negative rates.
    
    Args:
    - data_dist (DataDistribution): The data distribution instance.
    
    Returns:
    - tuple: Upper and lower boundary data for True Positive and True Negative rates.
    """
    thresholds = torch.linspace(0, 1, 100)
    upper_boundary = []
    lower_boundary = []
    
    for threshold in tqdm(thresholds, desc="Thresholds"):
        tp_upper, tn_upper = classifier_metrics(data_dist, threshold, upper=True)
        upper_boundary.append((tp_upper, tn_upper))

        tp_lower, tn_lower = classifier_metrics(data_dist, threshold, upper=False)
        lower_boundary.append((tp_lower, tn_lower))

    return upper_boundary, lower_boundary

class Oracle:
    def __init__(self, theta: float):
        """
        Initializes the oracle with a given theta for preference evaluation.
        
        Args:
        - theta (float): Oracle angle in radians.
        """
        self.theta = torch.tensor(theta)

    def evaluate_lpm(self, tp, tn):
        """
        Computes the linear performance metric (LPM) based on theta.
        
        Args:
        - tp (float): True Positive rate.
        - tn (float): True Negative rate.
        
        Returns:
        - float: Linear performance metric evaluation.
        """
        return torch.cos(self.theta) * tp + torch.sin(self.theta) * tn
    
    def preferred_classifier(self, tp_1, tn_1, tp_2, tn_2):
        """
        Determines the preferred classifier based on LPM values.
        
        Args:
        - tp_1, tn_1, tp_2, tn_2 (float): True Positive and True Negative rates for two classifiers.
        
        Returns:
        - bool: True if first classifier is preferred, False otherwise.
        """
        lpm_1 = self.evaluate_lpm(tp_1, tn_1)
        lpm_2 = self.evaluate_lpm(tp_2, tn_2)
        return (lpm_1 > lpm_2).item()
    
def theta_to_threshold(theta):
    """Converts theta angle to classification threshold."""
    return 1 / (1 + torch.tan(theta) ** -1)

def search_theta(oracle: Oracle, data_dist, lower_bound, upper_bound):
    """
    Performs a search over theta values to optimize the classification threshold.
    
    Args:
    - oracle (Oracle): The oracle for LPM evaluation.
    - data_dist (DataDistribution): The data distribution instance.
    - lower_bound (float): Lower bound for theta.
    - upper_bound (float): Upper bound for theta.
    
    Returns:
    - tuple: Updated lower and upper bounds for theta.
    """
    left = 0.75 * lower_bound + 0.25 * upper_bound
    middle = 0.5 * lower_bound + 0.5 * upper_bound
    right = 0.25 * lower_bound + 0.75 * upper_bound

    thetas = [lower_bound, left, middle, right, upper_bound]
    thresholds = theta_to_threshold(torch.tensor(thetas))
    new_lower, new_upper = None, None

    # YOUR CODE HERE (~18-25 lines)
    # 1. Collect metrics for each threshold value.
    # 2. Determine if LPM increases as theta increases.
    # 3. Check for pattern of increases and decreases in LPM.
    # 4. Update bounds based on observed LPM patterns.
    metrics = [classifier_metrics(data_dist, thresh) for thresh in thresholds]
    prefs = [oracle.preferred_classifier(*m2, *m1) for m1, m2 in zip(metrics, metrics[1:])]
    for i in reversed(range(len(prefs) - 1)):
        prefs[i] |= prefs[i + 1]
    new_lower, new_upper = lower_bound, upper_bound
    if not prefs[0] or not prefs[1]:
        new_upper = middle
    elif not prefs[2]:
        new_lower, new_upper = left, right
    else:
        new_lower = middle
    # END OF YOUR CODE

    return new_lower, new_upper

# Create instance and get upper & lower boundary data
data_dist = DataDistribution(N=10000000)
oracle = Oracle(theta=0.1)

def plot_confusion_region():
    """
    Plots the True Positive vs. True Negative rates for the upper and lower classifier boundaries.
    """
    upper_boundary, lower_boundary = sweep_classifiers(data_dist)

    # Prepare data for plotting for upper and lower boundaries
    tp_upper, tn_upper = zip(*upper_boundary)
    tp_lower, tn_lower = zip(*lower_boundary)

    # Plot the results for upper boundary
    plt.figure(figsize=(8, 6))
    plt.plot(tp_upper, tn_upper, marker='o', linestyle='-', alpha=0.7, label="Upper Boundary")
    plt.plot(tp_lower, tn_lower, marker='o', linestyle='--', alpha=0.7, label="Lower Boundary")
    plt.title("True Positive vs. True Negative Rates (Upper & Lower Boundaries)")
    plt.xlabel("True Positive Rate (TP)")
    plt.ylabel("True Negative Rate (TN)")
    plt.legend()
    plt.grid(True)
    plt.show()

def start_search():
    """
    Starts the theta search using the LPM-based oracle and prints the search range per iteration.
    """
    lower_bound = 0
    upper_bound = torch.pi / 2
    for _ in tqdm(range(10), desc="LPM Search"):
        print(f"Theta Search Space: [{lower_bound}, {upper_bound}]")
        lower_bound, upper_bound = search_theta(oracle, data_dist, lower_bound=lower_bound, upper_bound=upper_bound)
    print(f"Theta Search Space: [{lower_bound}, {upper_bound}]")
