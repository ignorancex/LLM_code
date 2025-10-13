LR = 1e-3
EPOCHS = 15000
BETAS = (0.9, 0.999)
EPSILON = 1e-8
CLIP_VALUE = 1.0

# Model initializations
EPS = 0.1
NUM_LAYERS_MLP = 2
C_IN = 4.0
C_OUT = 4.0
USE_ATT = True
USE_BIAS = True
DROPOUT = 0
TRAINING_CURVATURE = True
BATCH_SIZE = 256

# Cross Validation
TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.10

# Save weights
SAVE_PATH = None  # save weights here

# Number of features for graphs without initial node features such as COLLAB, REDDIT etc
NUM_NODE_FEATURES = 10
