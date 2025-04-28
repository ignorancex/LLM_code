import numpy as np

# True parameters of the model
parameters = {
    'A':{
        'display_name': '$A$',
        'val': 3
        },
    'B':{
        'display_name': '$B$',
        'val': 4
        },
    'g':{
        'display_name': '$g$',
        'val': 2
        },
    'k':{
        'display_name': '$k$',
        'val': 1
        }
    }

# Uniform prior specifications
priors = {
    'A':{
        'start': 0,
        'stop': 10
        },
    'B':{
        'start': 0,
        'stop': 10
        },
    'g':{
        'start': 0,
        'stop': 10
        },
    'k':{
        'start': 0,
        'stop': 10
        }
    }


# Number of statistics
nStats = 15
# Gamma true
gamma_true = np.array([1,1,1,1,0,0,0,0,0,0,0,0,0,0,0]).astype('bool')

# Number of replications
nR = 100