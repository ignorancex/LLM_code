import os
import sys

import numpy as np

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
                                                                          saving_reward=False)
    
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
    
    # folder name where reward model parameters are located ("results / reward_network_params / loading_folder_name")
    reward_loading_folder_name = constants.REWARD_LOADING_FOLDER
    reward_params_name = constants.REWARD_PARAMS_NAME

    # load pretrained reward network parameters if the pre-trained model is available
    reward_network = functions.load_reward_from_path(reward_network=reward_network,
                                                     results_path=results_path,
                                                     reward_loading_folder_name=reward_loading_folder_name,
                                                     reward_params_name=reward_params_name)
    # set model to evaluation mode
    for param in reward_network.parameters():
        param.requires_grad = False
    reward_network = reward_network.eval()
    
    # model optimizers and learning rate schedulers
    updater_obj = Updater(configs=configs,
                          policy_network=policy_network,
                          reward_network=reward_network)
    updater_obj.initialize_optimizers()
    
    # currently nu weight is zero; will be updated later
    nu_factor = torch.tensor(0.0)

    loss_policy_list = []
    
    print("\n================== Training Initialized (chuff chuff) ==================\n")

    for epoch in range(constants.BC_NUMBER_EPOCHS):

        # compute standar deviation minimization weight based on the annealing strategy
        std_weight = common.compute_annealing_factor(epoch=epoch,
                                                     max_epochs=constants.BC_NUMBER_EPOCHS)

        # loop through each batch inside the dataset
        for batch_train_data in tqdm(train_loader):

            # get batch of data
            input_state, output_action, _, _ = functions.read_each_loader(configs=configs,
                                                                          sample_data=tuple(batch_train_data))
            
            # forward pass to get mean of Gaussian distribution
            action, action_std, action_log_prob, action_entropy = policy_network.estimate_action(state=input_state,
                                                                                                 is_policy_inference=False,
                                                                                                 is_deterministic=True)
            
            # compute negative log-likelihood loss value for maximum likelihood estimation
            loss_bc_traj = updater_obj.multivariate_gaussian_nll_loss(action_true=output_action,
                                                                      action_pred_mu=action,
                                                                      action_log_std=torch.log(action_std),
                                                                      std_weight=std_weight)
            batch_loss = loss_bc_traj.mean().item()

            # backward pass and optimization for the policy model
            updater_obj.run_policy_optimizer(policy_loss=loss_bc_traj)
            
            loss_value_str = str(round(batch_loss, 5)).replace(".", "_")

        loss_policy_list.append(batch_loss)

        print(f"Epoch {epoch + 1}/{constants.BC_NUMBER_EPOCHS}, Batch Loss: {round(batch_loss, 5)}")

        if epoch % 10 == 0:
            functions.save_policy(epoch=epoch,
                                  policy_network=policy_network,
                                  saving_path=policy_saving_path,
                                  loss_value_str=str(abs(np.mean(loss_policy_list))).replace(".", "_"))
    
    print("\n================== Training Finished (choo choo) ==================\n")
