import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import torch

from tqdm import tqdm

# get the current script's directory
current_directory = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
# get the parent directory
parent_directory = os.path.dirname(current_directory)
# add the parent directory to the sys.path at the beginning
sys.path.insert(0, parent_directory)

from utils import constants, common
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
                                                                          saving_policy=True,
                                                                          saving_reward=True)
    
    # load train dataset of demonstrations
    training_data = PolicyDatasetLoader(demo_data_json_paths=json_paths_train)
    train_loader = torch.utils.data.DataLoader(training_data,
                                               batch_size=configs.batch_size_policy,
                                               shuffle=configs.data_shuffle,
                                               num_workers=configs.num_workers)

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
    activate_stopping = False
    is_early_stop = False
    early_stopping_counter = 0
    loss_reward_list, loss_policy_list, mean_reward_list = [], [], []

    # currently nu weight is zero; will be updated later
    nu_factor = torch.tensor(0.0)

    print("\n================== Training Initialized (chuff chuff) ==================\n")

    for epoch in range(constants.IRL_NUMBER_EPOCHS):

        print("================== Reward Training Phase ==================")

        # create episodes of trajectories by running policy model for data collection to refer to exploration-exploitation
        # take stochastic actions for behavior cloning by setting is_deterministic=False to sample actions from distribution
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
        mean_reward_list.append(np.mean(robot_reward_values))
        
        # convert robot execution trajectories to tensor format and stack them
        data_robo_tensor = functions.preprocess_trajectories(traj_list=robot_trajectories,
                                                             steps_tensor=data_robo_tensor,
                                                             is_demonstration=False)
        
        # randomly select a batch of data samples from the demonstration and robot trajectories
        selected_robo = np.random.choice(len(data_robo_tensor),
                                         int(configs.batch_size_reward),
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
        
        print("================== Policy Training Phase ==================")

        # compute standar deviation minimization weight based on the annealing strategy
        std_weight = common.compute_annealing_factor(epoch=epoch,
                                                     max_epochs=constants.IRL_NUMBER_EPOCHS)
        
        # loop through each batch inside the dataset
        for batch_train_data in tqdm(train_loader):

            # get batch of data
            input_state, output_action, _, _ = functions.read_each_loader(configs=configs,
                                                                          sample_data=tuple(batch_train_data))
            
            # forward pass to get mean of Gaussian distribution (take deterministic actions for behavior cloning)
            action, action_std, action_log_prob, action_entropy = policy_network.estimate_action(state=input_state,
                                                                                                 is_policy_inference=False,
                                                                                                 is_deterministic=True)
            
            # compute negative log-likelihood loss value for maximum likelihood estimation
            loss_bc_traj = updater_obj.multivariate_gaussian_nll_loss(action_true=output_action,
                                                                      action_pred_mu=action,
                                                                      action_log_std=torch.log(action_std),
                                                                      std_weight=std_weight)
            batch_loss = loss_bc_traj.mean().item()
            loss_value_str = str(round(batch_loss, 5)).replace(".", "_")

            # backward pass and optimization for the policy model
            updater_obj.run_policy_optimizer(policy_loss=loss_bc_traj)
        
        loss_policy_list.append(batch_loss)

        print(f"Epoch {epoch + 1}/{constants.IRL_NUMBER_EPOCHS}, Batch Policy Loss: {round(batch_loss, 5)}, Max-Ent Reward Loss: {round(maxent_loss.detach().item(), 5)}")
        
        if epoch % 2 == 0:
            functions.save_reward(epoch=epoch,
                                  reward_network=reward_network,
                                  saving_path=reward_saving_path,
                                  loss_value_str=str(abs(np.mean(loss_reward_list))).replace(".", "_"))
            functions.save_policy(epoch=epoch,
                                  policy_network=policy_network,
                                  saving_path=policy_saving_path,
                                  loss_value_str=str(abs(np.mean(loss_policy_list))).replace(".", "_"))
        
        if is_early_stop and activate_stopping:
            print(f"Early stopping at epoch {epoch + 1}!")
            break
    
    print("\n================== Training Finished (choo choo) ==================\n")

    plt.figure(figsize=[16, 12])
    plt.subplot(3, 1, 1)
    plt.title(f"Reward Function Max-Entropy Mean Loss Per Training Epoch")
    plt.plot(loss_reward_list, "r+")
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.title(f"Policy Model Gaussian Negative Log-Likelihood Batch Loss Per Training Epoch")
    plt.plot(loss_policy_list, "b+")
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.title(f"Average Reward on Robot Trajectory Per Training Epoch")
    plt.plot(mean_reward_list, "g+")
    plt.grid()

    plt.show()
