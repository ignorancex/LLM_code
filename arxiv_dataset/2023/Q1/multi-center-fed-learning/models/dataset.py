
DATASET_ATTRIBUTES = {
    
    'mnist':{
        'lr': 0.01,
        'dimension': 1000,
        'input_shape': (None, 28, 28),
    },
    
    'femnist':{
        'lr': 0.0003,
        'dimension': 1256,
        'input_shape': (None, 784),
    },
    
    'celeba': {
        'lr': 0.003,
        'dimension': 1152,
        'input_shape': (None, 84, 84, 3),
    },
    
    'cifar10': {
        'lr': 0.001,
        'dimension': 400,
        'input_shape': (None, 32, 32, 3),
    }
}
