import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
import gymnasium.spaces as spaces

NUM_CLASSES = 2
ACCURACIES = "accuracies"

class DataDistribution:
    def __init__(self, N: int, d: int = 10):
        """
        Initializes the data distribution with a specified number of samples.
        
        Args:
        - N (int): Number of data points.
        """
        # self.weights = torch.tensor([-0.3356, -1.4104, 0.3144, -0.5591, 1.0426, 0.6036, -0.7549, -1.1909, 1.4779, -0.7513])
        self.D = d
        gen = torch.Generator().manual_seed(42)
        self.weights = 3 * torch.rand((NUM_CLASSES, d), generator=gen) - 1.5
        self.data = torch.randn(N, self.D, generator=gen)
        self.probs = torch.softmax(self.data @ self.weights.T, axis=-1)
    
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

class Classifier:
    # For base implementation
    attributes_space = spaces.Dict({
        ACCURACIES: spaces.Box(0, 1, shape=(NUM_CLASSES,)),
        "speed": spaces.Box(0, 5, shape=(1,)),
        "cost": spaces.Box(-0.3, 0, shape=(1,)),
    })

    # For iteration plot experiment (note that NUM_CLASSES must be 2 here!)
    # attributes_space = spaces.Dict({
    #     ACCURACIES: spaces.Box(0, 1, shape=(NUM_CLASSES,)),
    #     "cost": spaces.Box(0, 10, shape=(1,)),
    # })

    # For 3 classes (NUM_CLASSES = 3) and 3 attributes 
    # attributes_space = spaces.Dict({
    #     ACCURACIES: spaces.Box(0, 1, shape=(NUM_CLASSES,)),
    #     "cost1": spaces.Box(-0.5, 0, shape=(1,)),
    #     "cost2": spaces.Box(0, 20, shape=(1,)),
    #     "cost3": spaces.Box(0, 30, shape=(1,)),
    # })

    def __init__(self, weights, wrong_class_prob=0, **properties):
        self.weights = weights
        assert all([k in Classifier.attributes_space.keys() for k in properties])
        self.properties = properties
        self.wrong_class_prob = wrong_class_prob

    def get_attributes(self, data_dist):
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
        predicted_class = F.one_hot(torch.argmax(data_dist.probs * self.weights, dim=-1), num_classes=NUM_CLASSES)
        accuracies = (probs * (1 - self.wrong_class_prob) * predicted_class).mean(axis=0)
        return {ACCURACIES: accuracies, **self.properties}

class Oracle:
    def __init__(self, a_star: dict):
        """
        Initializes the oracle with a given theta for preference evaluation.
        
        Args:
        - theta (float): Oracle angle in radians.
        """
        self.theta = {k: torch.tensor(v) for k, v in a_star.items()}

    def evaluate_dlpm(self, classifier_attributes):
        """
        Computes the linear performance metric (LPM) based on theta.
        
        Args:
        - tp (float): True Positive rate.
        - tn (float): True Negative rate.
        
        Returns:
        - float: Linear performance metric evaluation.
        """
        return sum([self.theta[k] @ classifier_attributes[k] for k in self.theta])
    
    def preferred_classifier(self, attr_1, attr_2):
        """
        Determines the preferred classifier based on LPM values.
        
        Args:
        - tp_1, tn_1, tp_2, tn_2 (float): True Positive and True Negative rates for two classifiers.
        
        Returns:
        - bool: True if first classifier is preferred, False otherwise.
        """
        dlpm_1 = self.evaluate_dlpm(attr_1)
        dlpm_2 = self.evaluate_dlpm(attr_2)
        return (dlpm_1 > dlpm_2).item()

def construct_test_classifier(m, test_attribute, test_attribute_dim, data_dist):
    properties = {attribute: torch.zeros(attribute_space.shape) for attribute, attribute_space in Classifier.attributes_space.items()}
    del properties[ACCURACIES]
    weights = torch.zeros(NUM_CLASSES)
    weights[0] = m
    if test_attribute == ACCURACIES:
        weights[test_attribute_dim] = 1 - m
        wrong_class_prob = 0
    else:
        test_attribute_space = Classifier.attributes_space[test_attribute]
        # introduce artificial tradeoff
        low, high = test_attribute_space.low[test_attribute_dim].item(), test_attribute_space.high[test_attribute_dim].item()
        theta = torch.atan(data_dist.probs[:, 0].mean()/(high - low) * m/(1 - m))
        properties[test_attribute][test_attribute_dim] = low + (high - low) * torch.cos(theta)
        wrong_class_prob = 1 - torch.sin(theta)
    return Classifier(weights, wrong_class_prob=wrong_class_prob, **properties)

def dlpm_elicitation(oracle, data_dist,  epsilon=0, max_iter=np.inf):
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
    num_queries = 0
    a_hat = {attribute: torch.zeros(attribute_space.shape) for attribute, attribute_space in Classifier.attributes_space.items()}
    a_hat[ACCURACIES][0] = 1
    for attribute in sorted(Classifier.attributes_space.keys()):
        attribute_space = Classifier.attributes_space[attribute]
        for attribute_dim in range(attribute_space.shape[0]):
            if attribute == ACCURACIES and attribute_dim == 0:
                continue

            # iterate over each axis to find appropriate ratio
            a = 0  # lower bound of binary search
            b = 1  # upper bound of binary search

            itr = 0
            while b - a > epsilon and itr < max_iter:
                c = (3 * a + b) / 4
                d = (a + b) / 2
                e = (a + 3 * b) / 4

                # get diagonal confusions for each point
                test_classifiers = [construct_test_classifier(x, attribute, attribute_dim, data_dist) for x in (a, c, d, e, b)]
                attributes = [test_classifier.get_attributes(data_dist) for test_classifier in test_classifiers]
                attr_a, attr_c, attr_d, attr_e, attr_b = attributes

                # query oracle for each pair
                response_ac = oracle.preferred_classifier(attr_a, attr_c)
                response_cd = oracle.preferred_classifier(attr_c, attr_d)
                response_de = oracle.preferred_classifier(attr_d, attr_e)
                response_eb = oracle.preferred_classifier(attr_e, attr_b)

                num_queries += 4


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
                itr += 1

            midpt = (a + b) / 2
            a_hat[attribute][attribute_dim] = (1 - midpt) / midpt
    print(f"Total number of queries: {num_queries}")
    one_norm = sum([v.sum().item() for v in a_hat.values()])
    return {k: v/one_norm for k, v in a_hat.items()}

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
def base_implementation():
    a_star = {ACCURACIES: [0.10, 0.05], "speed": [0.05], "cost": [0.80]}
    data_dist = DataDistribution(N=10000000)
    oracle = Oracle(a_star)
    a_hat = dlpm_elicitation(oracle, data_dist, epsilon=1e-3)
    print("A_hat", {k: a_hat[k] for k in sorted(a_hat)})
    print("A_star", {k: a_star[k] for k in sorted(a_star)})

def increased_class_and_attr():
    a_star = {ACCURACIES: [0.12, 0.08, 0.07], "cost1": [0.32], "cost2": [0.19], "cost3": [0.22]}
    data_dist = DataDistribution(N=10000000)
    oracle = Oracle(a_star)
    a_hat = dlpm_elicitation(oracle, data_dist, epsilon=1e-3)
    print("A_hat", {k: a_hat[k] for k in sorted(a_hat)})
    print("A_star", {k: a_star[k] for k in sorted(a_star)})

def create_iteration_plot():
    a_star = {ACCURACIES: [0.15, 0.05], "cost": [0.8]}
    data_dist = DataDistribution(N=10000000)
    oracle = Oracle(a_star)

    ax = plt.figure().add_subplot(projection='3d')
    ax.set_xlabel("Class 1 True Positive Weight")
    ax.set_ylabel("Class 2 True Positive Weight")
    ax.set_zlabel("Cost Weight")
    x = []
    y = []
    z = []
    for i in range(5):
        a_hat = dlpm_elicitation(oracle, data_dist, max_iter=i)
        x.append(a_hat["accuracies"][0].item())
        y.append(a_hat["accuracies"][1].item())
        z.append(a_hat["cost"].item())
        print(a_hat)
    
    for i in range(len(x)):
        if i == 0:
            ax.text(x[i]-0.01, y[i]-0.07, z[i]-0.01, f'iter {i}', color='black', fontsize=10)
        else:
            ax.text(x[i]+0.005, y[i]+0.005, z[i]+0.005, f'iter {i}', color='black', fontsize=10)
    ax.text(0.15-0.005, 0.05-0.04, 0.8-0.04, f'goal', color='red', fontsize=10)
    ax.plot3D(x,y,z)
    ax.scatter(x,y,z)
    ax.scatter([0.15],[0.05],[0.8],color="red")
    plt.show()

def plot_L1_error():
    a_star = {ACCURACIES: [0.10, 0.05], "speed": [0.05], "cost": [0.80]}
    a_star_values = np.array([0.10,0.05,0.05,0.80])
    data_dist = DataDistribution(N=10000000)
    oracle = Oracle(a_star)
    iterations = [i for i in range(7)]
    error = []
    for i in range(7):
        a_hat = dlpm_elicitation(oracle, data_dist, max_iter=i)
        a_hat_values = []
        a_hat_values.append(a_hat["accuracies"][0].item())
        a_hat_values.append(a_hat["accuracies"][1].item())
        a_hat_values.append(a_hat["speed"].item())
        a_hat_values.append(a_hat["cost"].item())
        a_hat_values = np.array(a_hat_values)
        error.append(np.mean(np.abs(a_hat_values - a_star_values)))
    plt.xlabel('Number of Iterations', fontsize=14)
    plt.ylabel('L1 Error of Weights', fontsize=14)
    plt.tick_params(axis='both', labelsize=12)
    plt.plot(iterations, error, linewidth=2)
    plt.show()

if __name__ == "__main__":
    base_implementation()
    # increased_class_and_attr()
    # base_implementation()
    # create_iteration_plot()
    # plot_L1_error()