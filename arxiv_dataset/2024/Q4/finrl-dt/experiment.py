import argparse
import loralib as lora
import numpy as np
import os
import pandas as pd
import pickle
import random
import torch

from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.evaluation.evaluate_episodes import (
    evaluate_episode_rtg, # this is for decision transformer with reward to go (rtg) - like DT + LLM + LoRA setup
    evaluate_episode # this is for simple MLP behavior cloning by Supervised Learning - no reward to go (rtg)
)
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.training.seq_trainer import SequenceTrainer # this is for decision transformer with reward to go (rtg) - like DT + LLM + LoRA setup
from decision_transformer.training.act_trainer import ActTrainer # this is for simple MLP behavior cloning by Supervised Learning
from utils import get_optimizer

# to make stock trading enviornment - we use finrl library
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.config import INDICATORS 

# prep for evaluation env. We need trade_data.csv for such prep of environment.
test_df = pd.read_csv('test_data.csv')
test_df = test_df.set_index(test_df.columns[0]) # this is to make sure that the index is not treated as a column
test_df.index.names = [''] # this is to make sure that the index is not treated as a column

# prep for training env. We need trade_data.csv for such prep of environment
train_df = pd.read_csv('train_data.csv')
train_df = train_df.set_index(train_df.columns[0]) # this is to make sure that the index is not treated as a column
train_df.index.names = [''] # this is to make sure that the index is not treated as a column

stock_dimension = len(test_df.tic.unique()) # this is to get the number of unique stocks in the trade data
state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension # this is to get the state space; 1 is for initial account value, 2*stock_dimension is for the stock prices and holdings, len(INDICATORS)*stock_dimension is for the technical indicators
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

buy_cost_list = sell_cost_list = [0.001] * stock_dimension # this follows the FinRL neurips 2018 paper setup
num_stock_shares = [0] * stock_dimension # this is to initialize the number of stock shares to 0 for all stocks

env_kwargs = {
    "hmax": 100, # holdings maximum - to limit the maximum number of shares that can be bought or sold at a time
    "initial_amount": 1000000, # initial account value (cash holdings)
    "num_stock_shares": num_stock_shares, # number of stock shares to hold
    "buy_cost_pct": buy_cost_list, # buying cost percentage
    "sell_cost_pct": sell_cost_list, # selling cost percentage
    "state_space": state_space, # state space
    "stock_dim": stock_dimension, # stock dimension
    "tech_indicator_list": INDICATORS, # technical indicators
    "action_space": stock_dimension, # action space
    "reward_scaling": 1e-4 # reward scaling 
}

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum

def experiment(
    variant, 
):  
    torch.manual_seed(variant["seed"])
    np.random.seed(variant["seed"])
    random.seed(variant["seed"])
    os.makedirs(variant["outdir"], exist_ok=True)
    device = variant.get("device", "cuda")
    env_name = variant["env"]
    
    if env_name == "stock_trading":
        # env for training
        env_train = StockTradingEnv(df = train_df, turbulence_threshold = 70, risk_indicator_col='vix', **env_kwargs)           
        train_trajectory_file = variant["dataset_path"]
        with open(train_trajectory_file, "rb") as f:
            train_trajectory = pickle.load(f)

        env_targets_train_reward = 0
        for i in range(len(train_trajectory)):
            env_targets_train_reward += train_trajectory[i]['rewards'].sum() / env_kwargs["reward_scaling"]
        env_targets_train = [env_targets_train_reward]
        max_ep_len_train = len(train_df)//stock_dimension

        # env for testing
        env_test = StockTradingEnv(df = test_df, turbulence_threshold = 70, risk_indicator_col='vix', **env_kwargs)
        test_trajectory_file = variant["test_trajectory_file"]
        with open(test_trajectory_file, "rb") as f:
            test_trajectory = pickle.load(f)
        max_ep_len = len(test_df)//stock_dimension
        env_targets_test = [2_000_000] # this is for evaluation of the trained DT because we need a RTG to make an inference 
        scale = env_kwargs["reward_scaling"]
    else:
        raise NotImplementedError
    
    if variant["model_type"] == "bc": # this is for simple MLP behavior cloning by Supervised Learning - no reward to go (rtg)
        env_targets = [0]  # since BC does not use rtg, no need for varying rtgs
    state_dim_train = env_train.observation_space.shape[0]
    act_dim_train = env_train.action_space.shape[0]
    print("act_dim_train:", act_dim_train)
    print("state_dim_train:", state_dim_train)

    state_dim = env_test.observation_space.shape[0]
    act_dim = env_test.action_space.shape[0]
    print("act_dim:", act_dim)
    print("state_dim:", state_dim)

    if env_name == "stock_trading":
        dataset_path = variant["dataset_path"] # this is a pickled file containing trajectories of stock trading as a list of dictionaries with keys: observations, actions, rewards, and terminals.
    else: 
        raise NotImplementedError
    
    with open(dataset_path, "rb") as f:
        trajectories = pickle.load(f)
    
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        states.append(path["observations"])
        traj_lens.append(len(path["observations"]))
        returns.append(-path["rewards"].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
    variant["state_mean"], variant["state_std"] = state_mean, state_std

    num_timesteps = sum(traj_lens)

    print("=" * 50)
    print(f"Starting new experiment: {env_name}")
    print(f"{len(traj_lens)} trajectories, {num_timesteps} timesteps found")
    print(f"Average return: {np.mean(-returns):.2f}, std: {np.std(returns):.2f}")
    print(f"Max return: {np.max(-returns):.2f}, min: {np.min(-returns):.2f}")
    print("=" * 50)

    K = variant["K"]
    batch_size = variant["batch_size"]
    pct_traj = variant.get("pct_traj", 1.0)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj * num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] < num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    print("num_trajectories: ", num_trajectories)
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    def get_batch(batch_size=256, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj["rewards"].shape[0] - 1)

            # get sequences from dataset
            s.append(traj["observations"][si : si + max_len].reshape(1, -1, state_dim))
            a.append(traj["actions"][si : si + max_len].reshape(1, -1, act_dim))
            r.append(traj["rewards"][si : si + max_len].reshape(1, -1, 1))
            if "terminals" in traj:
                d.append(traj["terminals"][si : si + max_len].reshape(1, -1))
            else:
                d.append(traj["dones"][si : si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = (
                max_ep_len - 1
            )  # padding cutoff
            rtg.append(
                discount_cumsum(traj["rewards"][si:], gamma=1.0)[
                    : s[-1].shape[1] + 1
                ].reshape(1, -1, 1)
            )
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate(
                [np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1
            )
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate(
                [np.ones((1, max_len - tlen, act_dim)) * -10.0, a[-1]], axis=1
            )
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = (
                np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1)
                / scale
            )
            
            timesteps[-1] = np.concatenate(
                [np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1
            )
            mask.append(
                np.concatenate(
                    [np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1
                )
            )

        if variant["fp16"] == True:
            float_dtype = torch.float16
        else:
            float_dtype = torch.float32
        
        s = torch.from_numpy(np.concatenate(s, axis=0)).to(
            dtype=float_dtype, device=device
        )
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(
            dtype=float_dtype, device=device
        )
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(
            dtype=float_dtype, device=device
        )
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(
            dtype=torch.long, device=device
        )
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(
            dtype=float_dtype, device=device
        )
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(
            dtype=torch.long, device=device
        )
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, d, rtg, timesteps, mask

    def eval_episodes_test(target_raw):
        def fn(model):
            returns, lengths = [], []     
            with torch.no_grad():
                if variant["model_type"] == "dt":
                    ret, length = evaluate_episode_rtg(
                        env_test,
                        state_dim,
                        act_dim,
                        model,
                        max_ep_len=max_ep_len,
                        scale=scale,
                        target_return=target_raw / scale,
                        target_reward_raw=target_raw,
                        state_mean=state_mean,
                        state_std=state_std,
                        device=device,
                        variant=variant,
                        eval_trajectory=test_trajectory,
                        train_or_test='test',
                    )
                elif variant["model_type"] == "bc":
                    ret, length = evaluate_episode(
                        env_test,
                        state_dim,
                        act_dim,
                        model,
                        max_ep_len=max_ep_len,
                        target_return=target_raw / scale,
                        state_mean=state_mean,
                        state_std=state_std,
                        device=device,
                        variant=variant,
                        eval_trajectory=test_trajectory,
                        train_or_test='test',
                    )

            returns.append(ret)
            lengths.append(length)

            return {
                f"target_{target_raw}_return_mean": np.mean(returns),
                f"target_{target_raw}_return_std": np.std(returns),
                f"target_{target_raw}_length_mean": np.mean(lengths),
                f"target_{target_raw}_length_std": np.std(lengths),
                f"target_{target_raw}_videos": []
            }

        return fn
    
    def eval_episodes_train(target_raw):
        print("running eval_episodes_train...")
        def fn(model):
            returns, lengths = [], []     
            with torch.no_grad():
                if variant["model_type"] == "dt":
                    ret, length = evaluate_episode_rtg(
                        env_train,
                        state_dim_train,
                        act_dim_train,
                        model,
                        max_ep_len=max_ep_len_train,
                        scale=scale,
                        target_return=target_raw / scale,
                        target_reward_raw=target_raw,
                        state_mean=state_mean,
                        state_std=state_std,
                        device=device,
                        variant=variant,
                        eval_trajectory=train_trajectory,
                        train_or_test='train',
                    )
                    
                elif variant["model_type"] == "bc":
                    ret, length = evaluate_episode(
                        env_train,
                        state_dim_train,
                        act_dim_train,
                        model,
                        max_ep_len=max_ep_len_train,
                        target_return=target_raw / scale,
                        state_mean=state_mean,
                        state_std=state_std,
                        device=device,
                        variant=variant,
                        eval_trajectory=train_trajectory,
                        train_or_test='train',
                    )

            returns.append(ret)
            lengths.append(length)

            return {
                f"target_{target_raw}_return_mean": np.mean(returns),
                f"target_{target_raw}_return_std": np.std(returns),
                f"target_{target_raw}_length_mean": np.mean(lengths),
                f"target_{target_raw}_length_std": np.std(lengths),
                f"target_{target_raw}_videos": []
            }

        return fn

    if variant["model_type"] == "dt":
        print("Initializing decision transformer model with some adapters...")
        model = DecisionTransformer(
            args=variant,
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len_train,
            hidden_size=variant["embed_dim"],
            n_layer=variant["n_layer"],
            n_head=variant["n_head"],
            n_inner=4 * variant["embed_dim"],
            activation_function=variant["activation_function"],
            n_positions=1024,
            resid_pdrop=variant["dropout"],
            attn_pdrop=0.1,
            mlp_embedding=variant["mlp_embedding"]
        )
        print("adapt mode: ", variant["adapt_mode"])
        print("adapt lora: ", variant["lora"])
        print("adapt embed: ", variant["adapt_embed"])

        if variant["adapt_mode"]: # we adapt or pretrained llm with some specifications
            print("adapt mode is true.. let's adapt our vanilla transformer model with some adapters.")
            if variant["lora"] == False:
                for param in model.parameters():
                    param.requires_grad = False
            else:
                print("adding lora adapter.")
                lora.mark_only_lora_as_trainable(model, bias='lora_only')
                print("lora is added. and only lora is trainable.")

            if variant["adapt_wte"]:
                print("adapt wte.")
                for param in model.transformer.wte.parameters():
                    param.requires_grad = True

            if variant["adapt_wpe"]:
                print("adapt wpe.")
                for param in model.transformer.wpe.parameters():
                    param.requires_grad = True

            if variant["adapt_embed"]:
                print("adapt embeddings.")
                # adapt the embeddings in DecisionTransformer
                for name, param in model.named_parameters():
                    if ("embed" in name or "predict" in name):
                        param.requires_grad = True
            if variant["adapt_ln"]:
                print("adapt layer norms.")
                # adapt the LayerNorm in the transformer's blocks
                for block in model.transformer.h:
                    for param in block.ln_1.parameters():
                        param.requires_grad = True
                    for param in block.ln_2.parameters():
                        param.requires_grad = True
                # adapt the final LayerNorm in the transformer
                for param in model.transformer.ln_f.parameters():
                    param.requires_grad = True
            if variant["adapt_attn"]:
                print("adapt attention.")
                for block in model.transformer.h:
                # adapt the attention weights and biases
                    for param in block.attn.parameters():
                        param.requires_grad = True
            if variant["adapt_ff"]:
                print("adapt feed-forward.")
                for block in model.transformer.h:
                    # adapt the feed_forward weights and biases
                    for param in block.mlp.parameters():
                        param.requires_grad = True
            if variant["only_adapt_last_two_blocks"]:
                print("for transformer, only adapt the last two blocks.")
                for block in model.transformer.h[0:-2]:
                    for param in block.parameters():
                        param.requires_grad = False
            if variant["adapt_last_two_blocks"]:
                print("for transformer, adapt the last two blocks.")
                for block in model.transformer.h[-2:]:
                    for param in block.parameters():
                        param.requires_grad = True
        else: 
            print("fintune all.")

    elif variant["model_type"] == "bc":
        print("Initializing behavior cloning model...")
        print("state_dim: ", state_dim)
        print("act_dim: ", act_dim)
        print("K: ", K)
        print("hidden_size: ", variant["embed_dim"])
        print("n_layer: ", variant["n_layer"])
        model = MLPBCModel(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            hidden_size=variant["embed_dim"],
            n_layer=variant["n_layer"],
        )

    else:
        raise NotImplementedError

    trainable_param_size = 0
    frozen_param_size = 0

    for name, param in model.named_parameters():
        if variant["model_type"] == "dt":
            if "transformer" not in name: continue
            if param.requires_grad:
                trainable_param_size += param.numel()
            else:
                frozen_param_size += param.numel()
        elif variant["model_type"] == "bc":
            if param.requires_grad:
                trainable_param_size += param.numel()
            else:
                frozen_param_size += param.numel()
                
    print(f"Trainable parameters: {trainable_param_size}")
    print(f"Frozen parameters: {frozen_param_size}")
    print(f"Trainable ratio: {trainable_param_size/(trainable_param_size + frozen_param_size)}")
    
    model = model.to(device=device)

    # # Generate natural language output from the LM with the given prompt
    # print("Generating text...!!!")
    # prompt = "Who's the president of the United States?"
    # print("prompt:", prompt)
    # device = next(model.parameters()).device
    # from transformers import GPT2Tokenizer
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # outputs = model.transformer_model.generate(**inputs)
    # generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print("Generated text..!!:", generated_text)
    # # END

    warmup_steps = variant["warmup_steps"]
    optimizer = get_optimizer(args=variant, model=model)
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda steps: min((steps + 1) / warmup_steps, 1)
    )
    
    visualize = variant["visualize"]

    if variant["model_type"] == "dt":
        trainer = SequenceTrainer(
            args=variant,
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
            eval_fns=[eval_episodes_train(tar) for tar in env_targets_train] + [eval_episodes_test(tar) for tar in env_targets_test],
        )
    elif variant["model_type"] == "bc":
        trainer = ActTrainer(
            args=variant,
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
            eval_fns=[eval_episodes_train(tar) for tar in env_targets_train] + [eval_episodes_test(tar) for tar in env_targets_test],
        )

    trainer.train_iteration(
        num_steps=variant["num_steps"]
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument("--env", type=str, default="Stock trading")
    # to load the dataset
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the trajectories.pkl file")
    parser.add_argument("--K", type=int, default=100)
    parser.add_argument("--pct_traj", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=64)
    # data sampling
    parser.add_argument("--sample_ratio", type=float, default=1.0)
    parser.add_argument("--data_suffix", type=str, default="d1")
    # training
    parser.add_argument("--device", type=str, default="cuda")
    # parser.add_argument("--log_to_wandb", "-w", action="store_true", default=False)
    parser.add_argument("--visualize", "-v", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=11102)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--fp16", action="store_true", default=False)
    # architecture, don't need to care about in our method
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=11) # this makes similar to decision transformer for comparison for behavior cloning
    parser.add_argument("--n_head", type=int, default=1)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--extend_positions", action="store_true", default=False)
    parser.add_argument("--share_input_output_proj", action="store_true", default=False)
    # learning hyperparameters
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", "-wd", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    # implementations
    parser.add_argument("--pretrained_lm", type=str, default=None)
    parser.add_argument("--mlp_embedding", action="store_true", default=False)
    # adaptations
    parser.add_argument("--adapt_mode", action="store_true", default=False)
    parser.add_argument("--lora", action="store_true", default=False)
    parser.add_argument("--only_adapt_last_two_blocks", action="store_true", default=False)
    parser.add_argument("--adapt_last_two_blocks", action="store_true", default=False)
    parser.add_argument("--adapt_ln", action="store_true", default=False)
    parser.add_argument("--adapt_attn", action="store_true", default=False)
    parser.add_argument("--adapt_ff", action="store_true", default=False)
    parser.add_argument("--adapt_embed", action="store_true", default=False)
    parser.add_argument("--adapt_wte", action="store_true", default=False)
    parser.add_argument("--adapt_wpe", action="store_true", default=False)    
    parser.add_argument("--random_weights_pretrained_lm", action="store_true", default=False)
    parser.add_argument("--exp_name", type=str, required=True, help="Name of the experiment")
    parser.add_argument("--drl_algo", type=str, required=True, help="Name of the DRL algorithm")
    parser.add_argument("--model_type", type=str, default="dt")  # dt for decision transformer, bc for behavior cloning 
    parser.add_argument("--num_steps", type=int, default=75500)
    parser.add_argument("--test_trajectory_file", type=str, default=None)

    args = parser.parse_args()
    print("args: ", vars(args))
    experiment(variant=vars(args))