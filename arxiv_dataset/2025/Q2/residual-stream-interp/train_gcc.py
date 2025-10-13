import os
import argparse
import numpy as np
import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from group_cc import GCC, L1Scheduler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--acts_dir', type=str)
    parser.add_argument('--ckpt_dir', type=str)
    parser.add_argument('--jumprelu', action='store_true')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--expansion', type=int, default=32)
    parser.add_argument('--topk', type=int, default=32)
    args = parser.parse_args()

    # print(os.listdir(args.acts_dir)[:3])
    # acts_ds = [np.load(os.path.join(args.acts_dir, path)) for path in os.listdir(args.acts_dir)[:3]]
    num_blocks = 1
    # acts_ds = np.concatenate(acts_ds, axis=1)
    # acts_ds = np.concatenate((acts_ds[0], acts_ds[1], acts_ds[2]), axis=1)
    # acts_ds = acts_ds[0]  #  SAE
    acts_data = np.load(args.acts_dir)
    acts_data = np.clip(acts_data, a_min=0, a_max=None)
    input_dims = acts_data[0].shape[-1]

    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    acts_data = torch.tensor(acts_data)
    acts_ds = TensorDataset(acts_data)
    dataloader = DataLoader(acts_ds, batch_size=args.batch_size, shuffle=True)

    gcc = GCC(input_dims, num_blocks, args.expansion, topk=args.topk if args.topk > 0 else None, auxk=256, dead_steps_threshold=64, jumprelu=args.jumprelu).to(device)
    if args.jumprelu:
        clipper = ThresholdClipper()
        gcc.apply(clipper)

    total_training_steps = args.epochs * len(dataloader)
    lr_warm_up_steps = total_training_steps // 5
    lr_decay_steps = total_training_steps // 5
    l1_warm_up_steps = total_training_steps // 2

    lr=0.0004
    adam_beta1=0.9 
    adam_beta2=0.999
    l1_lambda = 10
    lp_norm = 1
    aux_alpha = 10

    acts_sample = acts_data[torch.randint(0, acts_data.shape[0], (1024,))]
    mse_scale = (
        1 / ((acts_sample.float().mean(dim=0) - acts_sample.float()) ** 2).mean()
    )

    label = f'top{args.topk}_10aux_{args.expansion}exp' if args.topk > 0 else f'vanilla_{args.expansion}exp'

    # gcc.load_state_dict(torch.load(f"{args.ckpt_dir}/{label}_sae_weights_{10}ep.pth"))

    optimizer = optim.Adam(
        gcc.parameters(),
        lr=lr,
        betas=(
            adam_beta1,
            adam_beta2,
        ),
        eps=6.25e-10
    )

    # schedulers = [optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda steps: 1.0)]
    # schedulers.append(optim.lr_scheduler.LinearLR(
    #     optimizer,
    #     start_factor=1,
    #     end_factor=0,
    #     total_iters=lr_decay_steps,
    # ))
    # milestones = [total_training_steps - lr_decay_steps]
    # lr_scheduler = optim.lr_scheduler.SequentialLR(schedulers=schedulers, optimizer=optimizer, milestones=milestones)

    # lr_warmup = optim.lr_scheduler.LinearLR(
    #     optimizer,
    #     start_factor=(1/3),
    #     end_factor=1,
    #     total_iters=lr_warm_up_steps,
    # )

    # l1_scheduler = L1Scheduler(
    #     l1_warm_up_steps=l1_warm_up_steps,
    #     total_steps=total_training_steps,
    #     final_l1_coefficient=l1_lambda
    # )

    all_mses = []
    mse = torch.nn.functional.mse_loss
    for ep in range(0, args.epochs):
        print(f"Epoch {ep}")
        total_mse = 0
        total_l1 = 0
        total_dead = 0
        for i, acts in enumerate(dataloader, 0):
            optimizer.zero_grad()

            acts = acts[0].to(device)
            # acts = torch.clamp(acts, min=0)

            #   Group/trans coder
            # features, acts_hat = gcc(acts[:, :input_dims])
            # mse_loss = torch.nn.functional.mse_loss(acts_hat, acts[:, input_dims:], reduction="none").sum(-1)

            #   SAE
            features, acts_hat, dead_acts_recon, num_dead = gcc(acts)

            mse_loss = mse_scale * mse(acts_hat, acts, reduction="none").sum(-1)
            weighted_feature_acts = features * gcc.W_dec.norm(dim=1)
            sparsity = weighted_feature_acts.norm(p=lp_norm, dim=-1)  # sum over the feature dimension
            # l1_loss = l1_scheduler.current_l1_coefficient * sparsity
            # l1_loss = l1_lambda * sparsity

            total_mse += mse_loss.mean().cpu().detach().numpy()
            total_l1 += sparsity.mean().cpu().detach().numpy()
            total_dead += num_dead.cpu().numpy()

            # loss = mse_loss.mean()

            #  Auxiliary loss (prevents dead latents)
            # if dead_acts_recon is not None:
            error = acts - acts_hat
            aux_mse = mse(dead_acts_recon, error, reduction="none").sum(-1) / mse(error.mean(dim=0)[None, :].broadcast_to(error.shape), error, reduction="none").sum(-1)
            aux_loss = aux_alpha * aux_mse.nan_to_num(0)#.mean()
            # print(aux_loss)

            # loss += aux_loss

            loss = (mse_loss + aux_loss).mean()
            loss.backward()

            optimizer.step()
            # lr_warmup.step()
            # l1_scheduler.step()

        print(f"Average mse:  {total_mse / i}")
        all_mses.append(total_mse / i)

        print(f"Average L1:  {total_l1 / i}")

        print(f"Average Dead: {total_dead / i}")

        if (ep+1) % 10 == 0:
            torch.save(gcc.state_dict(), f"{args.ckpt_dir}/{label}_sae_weights_{ep+1}ep.pth")

        #   TODO:  validate by running on imnet val.  Record just the total reconstruction loss (omit sparsity loss).  Compare jumprelu vs vanilla.
        
    np.save(f"{args.ckpt_dir}/{label}_sae_maes_{ep+1}.npy", np.array(all_mses))