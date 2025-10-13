EPOCHS = 50
BATCH_SIZE = 64

BCE_STARTING_WEIGHT = 1
BCE_WEIGHT_DECAY = 1

KICK_IN_EPOCH = 15
KICK_IN_EPOCH2 = 25

CCL_STARTING_WEIGHT = 0.01

ORDER = [1, 2, 3, 4, 5]  # [1, 2, 3, 4, 5]

I = 0

config = {
    0: {
        'CCL_DECAY_RATE': 1.05,
        'FC1': 256,
        'FC2': 8,
        'SEPERATE_RUN': True,
        'OUTPUTDIM': 400
    },   
}