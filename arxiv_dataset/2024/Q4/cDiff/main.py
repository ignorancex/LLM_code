import torch
import torch.optim as optim
import numpy as np
from datasets import load_dataset
from models.neural_sampler import NormalizingFlowPosteriorSampler, DiffusionPosteriorSampler
from evaluation.SBC import sample_sbc_calstats, evaluate_sbc
from evaluation.TARP import get_ecp_area_difference
from utils import *
import pandas as pd
import time
import argparse

def trainer(data_loader, dataset, model, optimizer, scheduler, epochs, device, lr_decay, n_cal, L, seed, model_type, eval_interval, save_path):
    evaluation_sbc = pd.DataFrame()
    loss_record = []
    training_time_record = []
    evaluation_ecp = pd.DataFrame(columns=['epoch', 'inference_time', 'ecp_score'])
    ecp_traj = pd.DataFrame()
    for epoch in range(epochs):
        start_time = time.time()
        epoch_loss = []

        for batch in data_loader:
            theta, y = batch
            if y.shape[1] == 1:
                y = y.squeeze(1)
            y = y.to(device)
            theta = theta.to(device)

            optimizer.zero_grad()
            loss = model.loss(x=theta, y=y).mean()
            epoch_loss.append(float(loss))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
            optimizer.step()

        if lr_decay:
            scheduler.step()

        dataset.reset_batch_sample_sizes()

        print(
            f"Epoch: {epoch + 1}/{epochs},",
            f"Loss: {np.mean(epoch_loss):.2f},",
            f"LR: {scheduler.get_last_lr()[0]:.4f}"
        )

        loss_record.append(np.mean(epoch_loss))
        training_time_record.append(time.time() - start_time)

        if epoch % eval_interval == 0:
            inference_start_time = time.time()

            # sbc_calstats = sample_sbc_calstats(dataset, n_cal, L, theta.shape[-1], model, device)
            # eval_df = evaluate_sbc(sbc_calstats, seed, epoch, model_type)
            # evaluation_sbc = pd.concat([evaluation_sbc, eval_df], ignore_index=True)
            #
            # inference_time = time.time() - inference_start_time
            #
            # # Compute ecp score and ecp trajectory
            # ecp_score, ecp, alpha = get_ecp_area_difference(dataset, model, device, n_sim=args.ecp_n_sim, n_samples=args.ecp_n_samples)
            #
            # # Save metrics in the new metrics_df DataFrame
            # evaluation_ecp = pd.concat([evaluation_ecp, pd.DataFrame({
            #     'epochs': [epoch],
            #     'inference_time': [inference_time],
            #     'ecp_score': [ecp_score],
            #     'seed': [seed],
            #     'model_type': [model_type]
            # })], ignore_index=True)
            #
            # # Record ecp trajectory
            # ecp_traj[f"{model_type}_epoch_{epoch}_seed_{seed}"] = ecp
            # ecp_traj.index = alpha

            save_model(model, save_path, epoch, seed, model_type)

    epochs = list(range(1, len(loss_record) + 1))
    df_loss = pd.DataFrame({
        'epochs': epochs,
        'loss': loss_record,
        'seed': seed,
        'model_type': model_type,
        'training_time': training_time_record
    })

    return model, evaluation_sbc, evaluation_ecp, df_loss, ecp_traj


def main(args):
    # Dataset paramaters
    n_batches = args.n_batches
    batch_size = args.batch_size

    # Model paramaters
    hidden_dim_summary_net = 32
    n_summaries = 256  # sufficient statistics for normal-gamma model
    DEVICE = args.device
    alpha = args.alpha
    use_encoder = bool(args.use_encoder)
    n_sample = None if use_encoder else 1

    # Opitimzer paramaters
    epochs = args.epochs
    lr = args.lr
    lr_decay = args.lr_decay

    # Evaluate paramaters
    n_cal, L, model_type = args.n_cal, args.L, args.model

    if args.nickname is not None:
        model_type += args.nickname

    n_run = args.n_run
    eval_interval = args.eval_interval

    dataset_generator, sample_theta, sample_data = load_dataset(args.dataset)

    if args.use_encoder:
        dl = dataset_generator(n_batches, batch_size, return_ds=False)
    else:
        dl = dataset_generator(n_batches, batch_size, n_sample=1, return_ds=False)
    theta, y = next(iter(dl))
    y_dim = y.shape[-1]
    theta_dim = theta.shape[1]

    for i in range(1,n_run+1):
        seed = i + args.seed_start
        SET_SEED(seed)

        if args.model == "NormalizingFlow":
            model = NormalizingFlowPosteriorSampler(y_dim=y_dim, x_dim=theta_dim, n_summaries=n_summaries,
                                       hidden_dim_decoder=hidden_dim_summary_net, n_flows_decoder=32, alpha=alpha, device=DEVICE,
                                       use_encoder=use_encoder, data_type=args.data_type).to(DEVICE)
        elif args.model == "Diffusion":
            if args.use_emperical_sigma:
                sigma_data = theta.std().item()
            else:
                sigma_data = 0.5
            num_hidden_layer = args.num_hidden_layer
            model = DiffusionPosteriorSampler(y_dim=y_dim, x_dim=theta_dim, n_summaries=n_summaries,num_hidden_layer=num_hidden_layer,
                                              device=DEVICE,use_encoder=use_encoder, data_type=args.data_type, sigma_data=sigma_data)

        else:
            raise NotImplementedError


        save_path = f"{args.save_path}/{args.dataset}"
        os.makedirs(save_path, exist_ok=True)
        dl, ds = dataset_generator(n_batches, batch_size, n_sample, return_ds=True)

        if args.load_model:
            model = load_torch_model(model, save_path, epochs, seed, model_type)
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr)
            optimizer_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

            # Training
            model, evaluation_sbc, evaluation_ecp, df_loss, ecp_traj = trainer(dl,ds,model,optimizer,optimizer_sched,epochs,DEVICE,lr_decay,n_cal, L, seed,
                                         model_type, eval_interval, save_path)

        ## Final Evaluation
        sbc_calstats = sample_sbc_calstats(ds, n_cal, L, theta_dim, model, DEVICE)
        eval_df = evaluate_sbc(sbc_calstats, seed, epochs, model_type)
        evaluation_sbc = pd.concat([evaluation_sbc, eval_df], ignore_index=True)
        
        evaluation_sbc_save_path = f"{save_path}/evaluation_sbc.csv"
        evaluation_ecp_save_path = f"{save_path}/evaluation_ecp.csv"
        df_loss_save_path = f"{save_path}/loss.csv"
        ecp_traj_save_path = f"{save_path}/ecp_traj.csv"
        safe_update(evaluation_sbc, evaluation_sbc_save_path)
        safe_update(evaluation_ecp, evaluation_ecp_save_path)
        safe_update(df_loss, df_loss_save_path)
        safe_update(ecp_traj, ecp_traj_save_path, axis=1)

        if args.save_model:
            save_model(model, save_path, epochs, seed, model_type)

        # plot_hist(sbc_calstats,save_path,seed,model_type)
        if args.dataset in ["socks", "species_sampling","dirichlet_laplace"]:
            plot_scatter(y.squeeze(1),theta,model,save_path,seed,model_type,DEVICE)

        if args.dataset == "cos":
            from datasets.cos import plot_posterior, sample_and_plot
            plot_posterior(y_observed = 0.5)
            sample_and_plot(0.5,model,save_path, DEVICE, model_type, sample_steps=100, seed=seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple example of argparse")

    ## Training parameters
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--model', type=str, default="NormalizingFlow", help="NormalizingFlow or Diffusion")
    parser.add_argument('--n_run', type=int, default=1, help="How many runs to repeat")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_decay', action='store_true',)
    parser.add_argument('--n_batches', type=int, default=2, help="Number of batches for an epoch")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--save_path', type=str, default="./test")
    parser.add_argument('--alpha', type=float, default=0.1, help="Parameter for normalizing flow to control Lipschitz constant.")
    parser.add_argument('--device', type=int, default=3)
    parser.add_argument('--use_encoder', action='store_true', help="Use summary network or not")
    parser.add_argument('--use_emperical_sigma', action='store_true', help="whether to set \sigma_data as empirical std of data, otherwise 0.5 as EDM")
    parser.add_argument('--num_hidden_layer',type=int, default=4, help="Number of hidden layers for diffusion model")

    ## Dataset parameters
    parser.add_argument('--dataset', type=str, default="dirichlet_multinomial", help="Please see all datasets name in datasets/__init__.py")
    parser.add_argument('--data_type', type=str, default="iid", help="iid or time")

    ## Evaluation parameters
    parser.add_argument('--n_cal', type=int, default=1000, help="Number of calibration for SBC")
    parser.add_argument('--L', type=int, default=100, help="Number of posterior samples per x for SBC, same notation with SBC paper")
    parser.add_argument('--ecp_n_sim', type=int, default=1000, help="Number of simulations for TARP")
    parser.add_argument('--ecp_n_samples', type=int, default=2000, help="Number of posterior samples per x for TARP")
    
    # Utility parameters
    parser.add_argument('--save_model', action='store_true', help="Use encoder or not")
    parser.add_argument('--load_model', action='store_true', help="Use encoder or not")
    parser.add_argument('--eval_interval', type=int, default=2)
    parser.add_argument('--nickname', type=str, default=None, help="Add a nickname to the save folder")
    parser.add_argument('--seed_start', type=int, default=0)

    args = parser.parse_args()
    main(args)


