import enum
TRIAL = False
SANITY_CHECK = False
VERBOSE = True
CONV_LAYER_FROZEN = False
BATCH_SIZE = 20
VALIDATION_STEPS = 50
PREDICTION_STEPS = 25
INPUT_SHAPE = (384, 384, 3)
SAVE_MODEL_WEIGHTS = True
if TRIAL:
    STEPS_PER_EPOCH = 4
    EPOCHS = 2
    STEPS_PER_EPOCH_MODEL_2 = 4
    EPOCHS_MODEL_2 = 6
    seeds = [1970, 1972]
else:
    STEPS_PER_EPOCH = 100
    EPOCHS = 30
    STEPS_PER_EPOCH_MODEL_2 = 40
    EPOCHS_MODEL_2 = 60
    seeds = [1970, 1972, 2008, 2019, 2020]
class ensemble_learning_type(enum.Enum):
    soft_voting = 1
    averaging = 2
    optimizing = 3
ENSEMBLE_LEARNING = ensemble_learning_type.optimizing

def print_constants():
    print('*************************************************************')
    print('SANITY_CHECK       = ', str(SANITY_CHECK))
    print('CONV_LAYER_FROZEN  = ', str(CONV_LAYER_FROZEN))
    print('SAVE_MODEL_WEIGHTS = ', str(SAVE_MODEL_WEIGHTS))
    print('TRIAL              = ', str(TRIAL))
    print('*************************************************************')
