shapes_in = {
    'mnist': (1, 28, 28),
    'cifar10': (3, 32, 32),
    'cifar10c': (3, 32, 32),
    'agnews': (70, 300),
    'spambase': (57, )
}

shapes_out = {
    'mnist': 10,
    'cifar10': 10,
    'cifar10c': 10,
    'agnews': 4,
    'spambase': 1,  # binary classification, 2 class but 1 output
}
