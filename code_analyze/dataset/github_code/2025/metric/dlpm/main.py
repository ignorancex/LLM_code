import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm

class DataDistribution:
    def __init__(self, N: int, k: int, d: int = 10):
        """
        Initializes the data distribution with a specified number of samples.
        
        Args:
        - N (int): Number of data points.
        """
        # self.weights = torch.tensor([-0.3356, -1.4104, 0.3144, -0.5591, 1.0426, 0.6036, -0.7549, -1.1909, 1.4779, -0.7513])
        self.D = d
        self.num_classes = k
        gen = torch.Generator().manual_seed(42)
        self.weights = 3 * torch.rand((k, d), generator=gen) - 1.5
        self.data = torch.randn(N, self.D, generator=gen)
        self.probs = torch.softmax(self.data @ self.weights.T, axis=-1)
    
def classifier_metrics(data_dist, weights):
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
    predicted_class = F.one_hot(torch.argmax(probs * weights, dim=-1), num_classes=data_dist.num_classes)
    accuracies = (probs * predicted_class).mean(axis=0)
    return accuracies
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
    def __init__(self, a_star: torch.Tensor):
        """
        Initializes the oracle with a given theta for preference evaluation.
        
        Args:
        - theta (float): Oracle angle in radians.
        """
        self.theta = a_star

    def evaluate_dlpm(self, accuracies):
        """
        Computes the linear performance metric (LPM) based on theta.
        
        Args:
        - tp (float): True Positive rate.
        - tn (float): True Negative rate.
        
        Returns:
        - float: Linear performance metric evaluation.
        """
        return self.theta @ accuracies
    
    def preferred_classifier(self, accuracies_1, accuracies_2):
        """
        Determines the preferred classifier based on LPM values.
        
        Args:
        - tp_1, tn_1, tp_2, tn_2 (float): True Positive and True Negative rates for two classifiers.
        
        Returns:
        - bool: True if first classifier is preferred, False otherwise.
        """
        dlpm_1 = self.evaluate_dlpm(accuracies_1)
        dlpm_2 = self.evaluate_dlpm(accuracies_2)
        return (dlpm_1 > dlpm_2).item()

def rbo_dlpm(m, k1, k2, k):
    """
    This constructs DLPM weights for the upper boundary of the
    restricted diagonal confusions, given a parameter m.
    This is equivalent to \nu(m; k1, k2)
    
    Inputs:
    - m: parameter (between 0 and 1) for the upper boundary
    - k1: first axis for this  face
    - k2: second axis for this face
    - k: number of classes
    Outputs:
    - DLPM weights for this point on the upper boundary
    """
    new_a = torch.zeros(k)
    new_a[k1] = m
    new_a[k2] = 1 - m
    return new_a

def dlpm_elicitation(oracle, data_dist, k, epsilon=1e-5, max_iter=np.inf):
    """
    Inputs:
    - epsilon: some epsilon > 0 representing threshold of error
    - oracle: some function that accepts 2 confusion matrices and
        returns true if the first is preferred and false otherwise
    - get_d: some function that accepts dlpm weights and returns 
        diagonal confusions
    - k: number of classes
    Outputs:
    - estimate for true DLPM weights
    """
    a_hat = torch.zeros(k)
    a_hat[0] = 1
    for i in range(1, k):
        # iterate over each axis to find appropriate ratio
        a = 0  # lower bound of binary search
        b = 1  # upper bound of binary search

        iter = 0
        while (b - a > epsilon) and iter < max_iter:
            c = (3 * a + b) / 4
            d = (a + b) / 2
            e = (a + 3 * b) / 4

            # get diagonal confusions for each point
            d_a, d_c, d_d, d_e, d_b = (classifier_metrics(data_dist, rbo_dlpm(x, 0, i, k)) 
                for x in [a, c, d, e, b])

            # query oracle for each pair
            response_ac = oracle.preferred_classifier(d_a, d_c)
            response_cd = oracle.preferred_classifier(d_c, d_d)
            response_de = oracle.preferred_classifier(d_d, d_e)
            response_eb = oracle.preferred_classifier(d_e, d_b)

            # update ranges to keep the peak
            if response_ac:
                b = d
            elif response_cd:
                b = d
            elif response_de:
                a = c
                b = e
            elif response_eb:
                a = d
            else:
                a = d
            iter += 1

        midpt = (a + b) / 2
        a_hat[i] = (1 - midpt) / midpt
    return a_hat / torch.sum(a_hat)

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

# Create instance and get upper & lower boundary data
k=3
a_star = torch.tensor([0.3, 0.2, 0.5])
data_dist = DataDistribution(N=10000000, k=k)
oracle = Oracle(a_star)
a_hat = dlpm_elicitation(oracle, data_dist, k)
print("A_hat", a_hat, a_hat/a_hat.sum())
print("A_star", a_star, a_star/a_star.sum())