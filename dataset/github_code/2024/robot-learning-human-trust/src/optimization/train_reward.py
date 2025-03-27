import os
import sys

import numpy as np

import torch

# get the current script's directory
current_directory = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
# get the parent directory
parent_directory = os.path.dirname(current_directory)
# add the parent directory to the sys.path at the beginning
sys.path.insert(0, parent_directory)

from utils import constants
from utils.dataset_loader import PolicyDatasetLoader

from optimization import functions
from optimization.updater import Updater

from environment.environment import RobotEnvironment
from environment.buffer import ReplayBuffer

from models.policy_model import RobotPolicy
from models.reward_model import RewardFunction


if __name__ == "__main__":

    # available training machine
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training Device: ", device)
    
    configs = functions.setup_config(device=device)

    # create and return preliminary base paths
    json_paths_train, results_path = functions.get_directories(parent_directory=parent_directory,
                                                               data_folder_name=constants.DEMO_COLLECTION_DATE)
    policy_saving_path, reward_saving_path = functions.create_directories(configs=configs,
                                                                          results_path=results_path,
                                                                          saving_policy=False,
                                                                          saving_reward=True)
    
    # load train dataset of demonstrations
    training_data = PolicyDatasetLoader(demo_data_json_paths=json_paths_train)

    # get all indice numbers where the new trajectory is initialized in the dataset
    trajectory_indices = functions.find_indices_of_trajectory_changes(dataset=training_data)

    # we need both robot policy and reward function models
    policy_network = RobotPolicy(state_size=configs.state_size,
                                 hidden_size=configs.hidden_size,
                                 out_size=configs.action_size,
                                 log_std_min=configs.policy_log_std_min,
                                 log_std_max=configs.policy_log_std_max,
                                 device=configs.device)
    reward_network = RewardFunction(state_size=configs.state_size,
                                    action_size=configs.action_size,
                                    hidden_size=configs.hidden_size,
                                    out_size=configs.reward_size,
                                    device=configs.device)
    
    # model optimizers and learning rate schedulers
    updater_obj = Updater(configs=configs,
                        policy_network=policy_network,
                        reward_network=reward_network)
    updater_obj.initialize_optimizers()

    # construct custom environment for reward function training
    env = RobotEnvironment()
    env.set_reward_network(reward_network)
    env.is_reward_inference = False

    # create a replay buffer class object
    replay_buffer = ReplayBuffer(capacity=configs.replay_buffer_capacity)

    # folder name where policy model parameters are located ("results / policy_network_params / loading_folder_name")
    policy_loading_folder_name = constants.POLICY_LOADING_FOLDER
    policy_params_name = constants.POLICY_PARAMS_NAME

    # load pretrained policy network parameters if the pre-trained model is available
    policy_network = functions.load_policy_from_path(policy_network=policy_network,
                                                     results_path=results_path,
                                                     policy_loading_folder_name=policy_loading_folder_name,
                                                     policy_params_name=policy_params_name)
    # set model to evaluation mode
    for param in policy_network.parameters():
        param.requires_grad = False
    policy_network = policy_network.eval()

    # initialize empty tensors for demonstration and sample trajectories
    demo_traj_list = []
    data_demo_tensor, data_robo_tensor = torch.tensor([]), torch.tensor([])

    # get list of demonstration trajectories for training and analysis
    for traj_start_index in range(len(trajectory_indices)):
        traj_df, _, _, _ = functions.get_estimated_rewards(configs=configs,
                                                           updater_obj=updater_obj,
                                                           data_loader=training_data,
                                                           policy_network=policy_network,
                                                           reward_network=reward_network,
                                                           trajectory_indices=trajectory_indices,
                                                           traj_start_index=traj_start_index,
                                                           is_inference_reward=True,
                                                           is_inference_policy=True,
                                                           is_deterministic=True)
        demo_traj_list.append(traj_df)
        del traj_df
    
    # convert demonstrations to tensor format and stack them
    data_demo_tensor = functions.preprocess_trajectories(traj_list=demo_traj_list,
                                                         steps_tensor=data_demo_tensor,
                                                         is_demonstration=True)

    # parameters for early stopping criteria
    is_early_stop = False
    early_stopping_counter = 0
    loss_reward_list = []

    # currently nu weight is zero; will be updated later
    nu_factor = torch.tensor(0.0)

    print("\n================== Training Initialized (chuff chuff) ==================\n")

    for epoch in range(constants.RF_NUMBER_EPOCHS):

        # create episodes of trajectories by running policy model for data collection to refer to exploration-exploitation
        robot_trajectories = [functions.generate_session(env=env,
                                                         t_max=constants.TRAJECTORY_SIZE,
                                                         updater_obj=updater_obj,
                                                         replay_buffer=replay_buffer,
                                                         policy_network=policy_network,
                                                         is_policy_inference=True,
                                                         is_policy_gradient_update=False,
                                                         is_deterministic=False) for _ in range(constants.EPISODES_ROBOT_RUN)]
        
        # objein list of all rewards from the robot trajectories generated
        robot_reward_values = [robot_reward.item() for robot_trajectory in robot_trajectories for robot_reward in robot_trajectory[3]]

        # convert robot execution trajectories to tensor format and stack them
        data_robo_tensor = functions.preprocess_trajectories(traj_list=robot_trajectories,
                                                             steps_tensor=data_robo_tensor,
                                                             is_demonstration=False)
        
        # randomly select a batch of data samples from the demonstration and robot trajectories
        selected_robo = np.random.choice(len(data_robo_tensor),
                                         int(configs.batch_size_reward / 2),
                                         replace=True)
        selected_demo = np.random.choice(len(data_demo_tensor),
                                         int(configs.batch_size_reward),
                                         replace=False)
        data_batch_robo = data_robo_tensor[selected_robo].clone().detach()
        data_batch_demo = data_demo_tensor[selected_demo].clone().detach()

        # similar to the work explained in Guided Cost Learning paper, merge the demonstration and robot trajectories
        data_batch_both = torch.cat((data_batch_demo, data_batch_robo),
                                    dim=0)
        data_batch_both_random = data_batch_both[
            torch.randperm(int(data_batch_both.size(0)))]
        
        # get the state and action vectors from the merged batch
        states_robot, log_probs_robot, actions_robot = data_batch_both_random[:, :3], data_batch_both_random[:, 3:4], data_batch_both_random[:, 4:]
        states_expert, actions_expert = data_batch_demo[:, :3], data_batch_demo[:, 4:]

        # estimate rewards for the state from the merged batch
        robo_rewards = reward_network.estimate_reward(state=states_robot.float(),
                                                      action=actions_robot.float(),
                                                      is_reward_inference=False)
        demo_rewards = reward_network.estimate_reward(state=states_expert.float(),
                                                      action=actions_expert.float(),
                                                      is_reward_inference=False)
        
        # calculate the maximum entropy loss for the current batch similar to the work in Guided Cost Learning paper
        maxent_loss = updater_obj.calculate_max_entropy_loss(demonstration_rewards=demo_rewards,
                                                             robot_rewards=robo_rewards,
                                                             log_probability=log_probs_robot)
        
        # backward pass and optimization for the reward model
        updater_obj.run_reward_optimizer(reward_loss=maxent_loss)

        # append the loss value to the list for visualization
        loss_reward_list.append(maxent_loss.detach().item())

        # check for early stopping
        if torch.mean(demo_rewards) >= 0.999:
            is_early_stop = True
        
        if epoch % 10 == 0:
            functions.save_reward(epoch=epoch,
                                  reward_network=reward_network,
                                  saving_path=reward_saving_path,
                                  loss_value_str=str(abs(np.mean(loss_reward_list))).replace(".", "_"))
        
        if is_early_stop:
            print(f"Early stopping at epoch {epoch + 1}!")
            break
    
    print("\n================== Training Finished (choo choo) ==================\n")
