import random

from typing import Union

from raise_utils.learners import FeedforwardDL
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


def get_random_hyperparams(options: dict) -> dict:
    """
    Get hyperparameters from options.
    """
    hyperparams = {}
    for key, value in options.items():
        if isinstance(value, list):
            hyperparams[key] = random.choice(value)
        elif isinstance(value, tuple):
            hyperparams[key] = random.randint(value[0], value[1])
    return hyperparams


def get_many_random_hyperparams(options: dict, n: int) -> list:
    """
    Get n hyperparameters from options.
    """
    hyperparams = []
    for _ in range(n):
        hyperparams.append(get_random_hyperparams(options))
    return hyperparams


def get_learner(learner: str, config: dict) -> Union[DecisionTreeClassifier, LogisticRegression, GaussianNB, FeedforwardDL]:
    """
    Get a learner from a config.
    """
    if learner == 'tree':
        return DecisionTreeClassifier(
            criterion=config['criterion'],
            max_depth=config['max_depth'],
            min_samples_split=config['min_samples_split'],
            min_samples_leaf=config['min_samples_leaf'],
            max_features=config['max_features']
        )
    elif learner == 'logistic':
        return LogisticRegression(
            penalty=config['penalty'],
            solver='saga',
            C=config['C']
        )
    elif learner == 'nb':
        return GaussianNB()
    elif learner == 'ff':
        return FeedforwardDL(
            n_units=config['n_units'],
            n_layers=config['n_layers'],
            n_epochs=100
        )
    else:
        raise ValueError(f'Unknown learner: {learner}')

