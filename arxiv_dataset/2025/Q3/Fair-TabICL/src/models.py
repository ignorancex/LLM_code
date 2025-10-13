from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from tabpfn import TabPFNClassifier
from tabicl import TabICLClassifier
from tabdpt import TabDPTClassifier
from sklearn.ensemble import GradientBoostingClassifier


def get_model(model_name):

    if model_name == "lr":
        return LogisticRegression(max_iter=1400)
    elif model_name == "gbm":
        return GradientBoostingClassifier()
    elif model_name == "mlp":
        MLPClassifier(hidden_layer_sizes=[32, 32], max_iter=1000)
    elif model_name == "tabpfn":
        return TabPFNClassifier()
    elif model_name == "tabicl":
        return TabICLClassifier()
    elif model_name == "tabdpt":
        return TabDPTClassifier()

    raise ValueError(
        f"Model {model_name} not properly implemented in get_model function"
    )
