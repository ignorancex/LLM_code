"""Configuration file for common models/experiments"""

MAIN_PARAMS = { 
    'femnist': {
        'small': (30, 10, 15),
        'medium': (100, 10, 30),
        'large': (400, 20, 5)
        },
    'shakespeare': {
        'small': (6, 2, 2),
        'medium': (8, 2, 2),
        'large': (20, 1, 2)
        },
    'celeba': {
        'small': (30, 10, 2),
        'medium': (100, 10, 2),
        'large': (400, 20, 2)
        }
}
"""dict: Specifies execution parameters (tot_num_rounds, eval_every_num_rounds, clients_per_round)"""

MODEL_PARAMS = {
    'femnist.cnn': (0.0003, 62), # lr, num_classes
    'femnist.cnn_prox': (0.0003, 0.01, 62), # lr, num_classes
    'shakespeare.stacked_lstm': (0.0003, 80, 80, 256), # lr, seq_len, num_classes, num_hidden
    'celeba.cnn': (0.1, 2), # lr, num_classes
    'celeba.cnn_prox': (0.1, 0.01, 2),
    'synthetic.cnn': (0.01, 5, 60)
}
"""dict: Model specific parameter specification"""

ACCURACY_KEY = 'accuracy'
BYTES_WRITTEN_KEY = 'bytes_written'
BYTES_READ_KEY = 'bytes_read'
LOCAL_COMPUTATIONS_KEY = 'local_computations'
NUM_ROUND_KEY = 'round_number'
NUM_SAMPLES_KEY = 'num_samples'
CLIENT_ID_KEY = 'client_id'


""" constant for tensorflow model shape """
KERNAL_WIDTH, KERNEL_HEIGHT, NUM_INPUT_CHANNEL, NUM_OUTPUT_CHANNEL = 0, 1, 2, 3