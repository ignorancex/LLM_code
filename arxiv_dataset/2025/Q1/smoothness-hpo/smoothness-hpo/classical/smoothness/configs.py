learner_configs = {
    'tree': {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 5, 10],
        'max_features': ['sqrt', 'log2', None],
    },
    'logistic': {
        'penalty': ['l1', 'l2', 'elasticnet'],
        'C': [0.1, 1, 10],
    },
    'nb': {},
    'ff': {
        'n_units': (2, 6),
        'n_layers': (3, 20)
    }
}