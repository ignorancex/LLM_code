# state vector normalization parameters
MAX_DISTANCE_TO_OBJECT = 3.0 # meters
MAX_DISTANCE_TO_TARGET = 3.0 # meters
MAX_DISTANCE_TO_GROUND = 3.0 # meters

# action vector normalization parameters
END_EFFECTOR_POSITION_RANGE_X = [-2.0, 2.0] # meters
END_EFFECTOR_POSITION_RANGE_Y = [-2.0, 2.0] # meters
END_EFFECTOR_POSITION_RANGE_Z = [-1.0, 2.0] # meters

# initial state generation parameters (normalized)
INITIAL_STATE_MEANS = [0.175, 0.5, 0.3]
INITIAL_STATE_VARIANCES = [0.075, 0.05, 0.1]
INITIAL_STATE_MIN_RANGES = [0.10, 0.45, 0.20]
INITIAL_STATE_MAX_RANGES = [0.25, 0.55, 0.40]

# state vector column names
STATE_COLUMNS = ["distance_to_object",
                 "distance_to_target",
                 "distance_to_ground"]

# action vector column names
ACTION_COLUMNS = ["arm_action_x",
                  "arm_action_y",
                  "arm_action_z",
                  "arm_rotation_x",
                  "arm_rotation_y",
                  "arm_rotation_z",
                  "arm_rotation_w"]

# behavior cloning policy model trajectories column names
COLUMN_NAMES = ["time_seconds"] + \
                STATE_COLUMNS + \
                ACTION_COLUMNS

# column names for normalized and denormalized state and action vectors
STATE_NORMALIZED_LABEL_NAME = "state_label_norm" # normalized state vector
STATE_DENORMALIZED_LABEL_NAME = "state_label_denorm" # denormalized state vector
STATE_ESTIMATION_NORMALIZED_NAME = "state_est_norm" # estimated normalized state vector
STATE_ESTIMATION_DENORMALIZED_NAME = "state_est_denorm" # estimated denormalized state vector
NEXT_STATE_NORMALIZED_LABEL_NAME = "next_state_label_norm" # normalized next state vector
NEXT_STATE_DENORMALIZED_LABEL_NAME = "next_state_label_denorm" # denormalized next state vector
NEXT_STATE_ESTIMATION_NORMALIZED_NAME = "next_state_est_norm" # estimated normalized state vector
NEXT_STATE_ESTIMATION_DENORMALIZED_NAME = "next_state_est_denorm" # estimated denormalized state vector
ACTION_NORMALIZED_LABEL_NAME = "action_label_norm" # normalized action label
ACTION_DENORMALIZED_LABEL_NAME = "action_label_denorm" # denormalized action vector
ACTION_PREDICTION_NAME = "action_pred_norm" # mean of distribution
ACTION_PREDICTION_DENORMALIZED_NAME = "action_pred_denorm" # x, y, z locations
ACTION_PREDICTION_STD_NAME = "action_pred_std" # standard deviation
ACTION_LABEL_LOGPROB_NAME = "action_label_logprob" # log probability of demonstration action
ACTION_LABEL_ENTROPY_NAME = "action_label_entropy" # entropy of demonstration action
ACTION_PREDICTION_LOGPROB_NAME = "action_pred_logprob" # log probability of action prediction
ACTION_PREDICTION_ENTROPY_NAME = "action_pred_entropy" # entropy of action prediction
ACTION_PREDICTION_AVG_LOGPROB_NAME = "action_pred_avg_logprob" # average log probability of actions
REWARD_DEMONSTRATION_TRAJECTORY_NAME = "reward_demo_traj" # reward of demonstration trajectory
REWARD_ROBOT_TRAJECTORY_NAME = "reward_robot_traj" # reward of robot estimated trajectory

# column name to represent number index of the state in a trajectory
STATE_NUMBER_COLUMN = "state_number"

# trajectory index number column name
NUMBER_TRAJECTORY_COLUMN = "trajectory_index"

# multivarate gaussian log likelihood loss value column name
GAUSSIAN_NLL_LOSS_COLUMN = "gnll_loss"

# random seed for reproducibility
RANDOM_SEED = 1773

# end-effector position is next action value for the given state (constant : -1)
ACTION_LABEL_SHIFT_IDX = -1

# number of state action pairs to discritize in each trajectory
TRAJECTORY_SIZE = 20 # equally spaced number of points

# number of episodes to run the robot for data collection at each epoch for exploration
EPISODES_ROBOT_RUN = 20

# collected dataset folder name in ("dataset \\ human_demonstrations \\ collection_date")
DEMO_COLLECTION_DATE = "2024_01_23_Train" # year_month_day NOTE: make sure that data date is correct
TEST_COLLECTION_DATE = "2024_02_02_Test" # year_month_day NOTE: make sure folder name is correct

# experiment outcomes folder name in ("dataset \\ robot_executions \\ experiment_date")
ROBOT_EXPERIMENT_DATE = "2024_02_23_Learning" # year_month_day NOTE: make sure that data date is correct
ROBOT_INFERENCE_DATE = "2024_02_23_Inference" # year_month_day NOTE: make sure that data date is correct

# collected experiments with robot should be saved in the following folders ("results \\ experiments \\ ___")
LEARNING_EXPERIMENT_FOLDER = "learning_experiments" # learning trust experiment results stored in this folder
INFERENCE_EXPERIMENT_FOLDER = "inference_experiments" # inference experiment results stored in this folder

# altitute of the base of the robot from the ground as a focal point
ROBOT_BASE_HEIGHT = 0.425 # meters NOTE: make sure the alignment with robot experiment

# experiment obstacle location x, y, z coordinates
OBSTACLE_LOCATION = [-0.794, 0.267, 0.129] # meters NOTE: make sure the alignment with experiment

# experiment target location x, y, z coordinates
TARGET_LOCATION = [-0.382, 0.792, -0.325] # meters NOTE: make sure the alignment with robot experiment

# number of training epoch for behavior cloning
BC_NUMBER_EPOCHS = 50

# number of training epoch for reward function
RF_NUMBER_EPOCHS = 100

# number of training epoch for IRL algorithm
IRL_NUMBER_EPOCHS = 150

# folder name where parameters are located ("results \\ policy_network_params \\ policy_loading_folder")
POLICY_LOADING_FOLDER = "Feb_23_2024-23_23_23" # year_month_day-hh_mm_ss NOTE: make sure that folder name is correct
POLICY_PARAMS_NAME = "policy_network_epoch_65_best_model.pt" # NOTE: make sure that file name exists

# folder name where parameters are located ("results \\ reward_network_params \\ reward_loading_folder")
REWARD_LOADING_FOLDER = "Feb_23_2024-23_23_23" # year_month_day-hh_mm_ss NOTE: make sure that folder name is correct
REWARD_PARAMS_NAME = "reward_network_epoch_99_loss_best_model.pt" # NOTE: make sure that file name exists
