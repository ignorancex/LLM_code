import numpy as np
import torch
import pickle
import os

def evaluate_episode(
    env,
    state_dim,
    act_dim,
    model,
    max_ep_len=1000,
    device="cuda",
    target_return=None,
    state_mean=0.0,
    state_std=1.0,
    variant=None,
    train_or_test='test',
    eval_trajectory=None,
):
    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = (
        # torch.from_numpy(state)
        torch.tensor(state[0]) # don't know why but state is a tuple of list of numbs and {}.
        .reshape(1, state_dim)
        .to(device=device, dtype=torch.float32)
    )
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32)
    episode_return, episode_length = 0, 0
    total_asset_value_list = [env.initial_amount]
    loss_list = []

    for t in range(max_ep_len):
        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return=target_return,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        # Calculate loss if eval_trajectory is provided
        if eval_trajectory is not None:
            expert_action = eval_trajectory[0]['actions'][t]
            loss = torch.mean((torch.from_numpy(expert_action) - torch.from_numpy(action)) ** 2)
            print("bc loss:", loss.item())
            loss_list.append(loss.item())

        state, reward, done, _, _ = env.step(action)

        total_asset_value_list.append(total_asset_value_list[-1] + reward * (1/env.reward_scaling))

        cur_state = torch.from_numpy(np.array(state)).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward





        episode_return += reward
        episode_length += 1

        if done:
            # Save data
            outdir = variant.get('outdir', 'output')
            os.makedirs(outdir, exist_ok=True)

            # Save total asset value change
            file_name = f'total_asset_value_change_{train_or_test}.pkl'
            file_path = os.path.join(outdir, file_name)
            with open(file_path, 'wb') as f:
                pickle.dump(total_asset_value_list, f)

            # Save loss list if available
            if loss_list:
                loss_list_file_name = f'{train_or_test}_loss_list.pkl'
                loss_list_file_path = os.path.join(outdir, loss_list_file_name)
                with open(loss_list_file_path, 'wb') as f:
                    pickle.dump(loss_list, f)

            break

    return episode_return, episode_length


def evaluate_episode_rtg(
    env,
    state_dim,
    act_dim,
    model,
    max_ep_len=1000,
    scale=1000.0,
    state_mean=0.0,
    state_std=1.0,
    device="cuda",
    target_return=None,
    target_reward_raw=None,
    eval_trajectory=None,
    variant=None,
    train_or_test='test',
):
    expert_actions = eval_trajectory[0]['actions']

    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()
    
    states = (
        torch.tensor(state[0]) # don't know why but state is a tuple of list of numbs and {}.
        .reshape(1, state_dim)
        .to(device=device, dtype=torch.float32)
    )

    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(
        1, 1
    )
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    episode_return, episode_length = 0, 0

    loss_list = []
    total_asset_value_list = [env.initial_amount]

    for t in range(max_ep_len):
        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )
        actions[-1] = action # update the last action in actions
        action = action.detach().cpu().numpy()

        expert_action = expert_actions[t]
        test_loss = torch.mean((torch.from_numpy(expert_action) - torch.from_numpy(action)) ** 2)
        loss_list.append(test_loss.item())
                
        state, reward, done, _, _ = env.step(action)
        print("reward:", reward)
        total_asset_value_list.append(total_asset_value_list[-1] + reward * (1/env.reward_scaling))

        cur_state = torch.from_numpy(np.array(state)).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        pred_return = target_return[0, -1]
        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)],
            dim=1,
        )

        episode_return += reward
        episode_length += 1
        
        if done:
            # current time
            from datetime import datetime
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            import pickle
            import os

            # Create the output directory if it doesn't exist
            outdir = variant.get('outdir', 'output')  # Default to 'output' if not specified
            os.makedirs(outdir, exist_ok=True)

            # Construct the file path
            file_name = f'total_asset_value_change_{train_or_test}_env_{target_reward_raw}.pkl'
            file_path = os.path.join(outdir, file_name)

            # Save the loss list
            loss_list_file_name = f'{train_or_test}_loss_list_{target_reward_raw}.pkl'
            loss_list_file_path = os.path.join(outdir, loss_list_file_name)
            with open(loss_list_file_path, 'wb') as f:
                pickle.dump(loss_list, f)

            # Save the pickle file
            with open(file_path, 'wb') as f:
                pickle.dump(total_asset_value_list, f)

            break

    model.past_key_values = None

    return episode_return, episode_length


