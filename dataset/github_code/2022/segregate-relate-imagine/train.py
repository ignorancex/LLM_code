import os
from pathlib import Path
import datetime
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from third_party.genesis_v2.geco import GECO
from third_party.genesis_v2.shapestacks_config import ShapeStacksDataset
from sri.visualization import visualise_outputs
from sri.model import SRI
import argparse

local_rank = os.environ['LOCAL_RANK']

def save_checkpoint(ckpt_file, model, optimiser, beta, geco_err_ema,
                    iter_idx, verbose=True):
    if verbose:
        print(f"Saving model training checkpoint to: {ckpt_file}")
    ckpt_dict = {'model_state_dict': model.state_dict(),
                'optimiser_state_dict': optimiser.state_dict(),
                'beta': beta[0],
                'beta_sri': beta[1],
                'err_ema': geco_err_ema[0],
                'err_sri_ema': geco_err_ema[1],
                'iter_idx': iter_idx}

    torch.save(ckpt_dict, ckpt_file)


def run(training, seed):
    global local_rank
    local_rank = f'cuda:{local_rank}'

    checkpoint_dir = Path(training['out_dir'], 'weights')
    tb_dir = Path(training['out_dir'], 'tb')
    for dir_ in [checkpoint_dir, tb_dir]:
        if local_rank == 'cuda:0' and not dir_.exists():
            os.makedirs(dir_)

    tb_dbg = tb_dir / (training['run_suffix'] + '_' + datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))

    if local_rank == 'cuda:0' and not training['debug']:
        print(f'Creating SummaryWriter! ({local_rank})')
        writer = SummaryWriter(tb_dbg)
        
    
    print(f'Local rank: {local_rank}. Setting random seed to {seed}')
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    np.random.seed(seed)
        
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(training['DDP_port'])
    torch.distributed.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)

    if training['dataset'] == 'shapestacks':
        tr_dataset = ShapeStacksDataset(data_dir=training['data_dir'], mode=training['mode'])
    else:
        raise ValueError(f'Unsupported dataset {training["dataset"]}')

    batch_size = training['batch_size']
    tr_sampler = DistributedSampler(dataset=tr_dataset)
    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    tr_dataloader = torch.utils.data.DataLoader(tr_dataset,
            batch_size=batch_size, sampler=tr_sampler, 
            num_workers=training['num_workers'],
            worker_init_fn=worker_init_fn,
            drop_last=True)

    model = SRI(training['K'], training['pixel_bound'], training['z_dim'], 
                training['semiconv'], training['kernel'], training['pixel_std'],
                training['dynamic_K'], training['image_likelihood'], training['debug'],
                training['img_size'], training['s_dim'], training['L'])
    num_elements = 3 * model.img_size**2  # Assume three input channels

    # Goal is specified per pixel & channel so it doesn't need to
    # be changed for different resolutions etc.
    geco_goal = training['g_goal'] * num_elements
    # Scale step size to get similar update at different resolutions
    geco_lr = training['g_lr'] * (64**2 / model.img_size**2)
    geco = GECO(geco_goal, geco_lr, training['g_alpha'], training['g_init'],
                training['g_min'], training['g_speedup'])
    beta = geco.beta

    geco_sri = GECO(geco_goal, geco_lr, training['g_alpha'], training['g_init'],
            training['g_min'], training['g_speedup'])
    beta_sri = geco_sri.beta
    
    model = model.to(local_rank)
    
    # Optimization
    model_opt = torch.optim.Adam(model.parameters(), lr=training['learning_rate'])
    geco.to_cuda()
    geco_sri.to_cuda()
    # Try to restore model and optimiser from checkpoint
    iter_idx = 0
    if training['load_from_checkpoint']:
        checkpoint = checkpoint_dir / training['checkpoint']
        map_location = {'cuda:0': local_rank}
        state = torch.load(checkpoint, map_location=map_location)
        model.load_state_dict(state['model_state_dict'], strict=True)
        model_opt.load_state_dict(state['optimiser_state_dict'])
        iter_idx = state['iter_idx'] + 1
        if 'beta' in state:
            geco.beta = state['beta']
        if 'err_ema' in state:
            geco.err_ema = state['err_ema']
        if 'beta_sri' in state:
            geco_sri.beta = state['beta_sri']
        if 'err_sri_ema' in state:
            geco_sri.err_ema = state['err_sri_ema']

        
        # Update starting iter
        iter_idx = state['iter_idx'] + 1
    print(f"Starting training at iter = {iter_idx}")

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank,
                                                      find_unused_parameters=False)
    model.train()
    if local_rank == 'cuda:0':
        print(model)

    # ------------------------
    # TRAINING
    # ------------------------
    model.train()
    epoch = 0
    
    while iter_idx <= training['iters']:
        
         # Re-shuffle every epoch
        tr_sampler.set_epoch(epoch)
        if training['tqdm'] and local_rank == 'cuda:0':
            data_iter = tqdm(tr_dataloader)
        else:
            data_iter = tr_dataloader
        for train_batch in data_iter:
            
            # Parse data
            train_input = train_batch['imgs']
            # back to 0-1
            train_input = (train_input + 1) / 2.
            train_input = train_input.to(local_rank)

            # Forward propagation
            model_opt.zero_grad()
            (output, sri_output), losses, stats, att_stats, comp_stats = model(train_input)
           
            # Reconstruction errors
            err = losses.genv2_err.mean(0)
            err_sri = losses.sri_err.mean(0)
            # KL divergences
            kl_m, kl_l = torch.tensor(0), torch.tensor(0)
                # -- KL stage 1
            if 'kl_m' in losses:
                kl_m = losses.kl_m.mean(0)
            elif 'kl_m_k' in losses:
                kl_m = torch.stack(losses.kl_m_k, dim=1).mean(dim=0).sum()
            # -- KL stage 2
            if 'kl_l' in losses:
                kl_l = losses.kl_l.mean(0)
            elif 'kl_l_k' in losses:
                kl_l = torch.stack(losses.kl_l_k, dim=1).mean(dim=0).sum()

            # -- KL scene
            kl_scene = losses.sri_kl_scene
            if kl_scene is not None:
                kl_scene = kl_scene.mean(0)
            # -- KL z ordered (scene composition)
            kl_rollout = losses.sri_kl_z_ordered.mean(0)
            # -- Imagined KL
            kl_slot = losses.sri_kl_slot.mean(0)

            # Compute ELBOs
            elbo = (err + kl_l + kl_m).detach()
            # Compute MSE / RMSE
            mse_batched = ((train_input-output)**2).mean((1, 2, 3)).detach()
            rmse_batched = mse_batched.sqrt()
            mse, rmse = mse_batched.mean(0), rmse_batched.mean(0)
            
            kl = kl_scene + kl_rollout + kl_slot
            sri_elbo = (err_sri + kl)
            
            # Main objective
            
            beta = geco.beta
            loss = geco.loss(err, kl_l + kl_m)
            
            beta_sri = geco_sri.beta
            loss_sri = geco_sri.loss(err_sri, kl)

            total_loss = loss + loss_sri

            # Backprop and optimise
            total_loss.backward()

            model_opt.step()
            
            # Heartbeat log
            if not training['debug'] and iter_idx % training['tensorboard_freq'] == 0 and iter_idx > 0 and local_rank == 'cuda:0':
                # TensorBoard logging
                # -- Optimisation stats
                writer.add_scalar('optim/beta', beta, iter_idx)
                writer.add_scalar('optim/geco_err_ema',
                                    geco.err_ema, iter_idx)
                writer.add_scalar('optim/geco_err_ema_element',
                                    geco.err_ema/num_elements, iter_idx)
                # -- Main loss terms
                writer.add_scalar('train/err', err, iter_idx)
                writer.add_scalar('train/err_element', err/num_elements, iter_idx)
                writer.add_scalar('train/kl_m', kl_m, iter_idx)
                writer.add_scalar('train/kl_l', kl_l, iter_idx)
                writer.add_scalar('train/elbo', elbo, iter_idx)
                writer.add_scalar('train/loss', loss, iter_idx)
                writer.add_scalar('train/mse', mse, iter_idx)
                writer.add_scalar('train/rmse', rmse, iter_idx)

                writer.add_scalar('optim/beta_sri', beta_sri, iter_idx)
                writer.add_scalar('optim/geco_err_sri_ema',
                                    geco_sri.err_ema, iter_idx)
                writer.add_scalar('train/err_sri', err_sri, iter_idx)
                if kl_scene is not None:
                    writer.add_scalar('train/kl_scene', kl_scene, iter_idx)
                writer.add_scalar('train/kl_rollout', kl_rollout, iter_idx)
                writer.add_scalar('train/kl_slot', kl_slot, iter_idx)
                writer.add_scalar('train/elbo_sri', sri_elbo, iter_idx)
                writer.add_scalar('train/loss_loss', loss_sri, iter_idx)

                # -- Per step loss terms
                for key in ['kl_l_k', 'kl_m_k']:
                    if key not in losses: continue
                    for step, val in enumerate(losses[key]):
                        writer.add_scalar(f'train_steps/{key}{step}',
                                          val.mean(0), iter_idx)

                # Visualise model outputs
                visualise_outputs(model, train_input, writer, 'train', iter_idx)

            if (iter_idx % training['checkpoint_freq'] == 0 and 
                local_rank == 'cuda:0'):
                # Save the model
                prefix = training['run_suffix']
                save_checkpoint(
                    checkpoint_dir / f'{prefix}-state-{iter_idx}.pth',
                    model.module, model_opt, [beta, beta_sri],
                    [geco.err_ema, geco_sri.err_ema], iter_idx, False)

            
            if iter_idx >= training['iters']:
                iter_idx += 1
                break
            iter_idx += 1
        epoch += 1

    # Close writer
    if local_rank == 'cuda:0' and not training['debug']:
        writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('segregate-reglate-imagine training')
    
    # training
    parser.add_argument('--DDP_port', type=int, default=29500,
                        help='torch.distributed config')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='training mini-batch size')
    parser.add_argument('--dataset', type=str, default='shapestacks',
                        help='the training dataset name')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of pytorch dataloader workers')
    parser.add_argument('--mode', type=str, default='train',
                        help='train/val/test')
    parser.add_argument('--data_dir', type=str, default='',
                        help='$PATH_TO/shapestacks')
    parser.add_argument('--iters', type=int, default=500000, 
                        help='number of training steps')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='learning rate for Adam')
    parser.add_argument('--tensorboard_freq', type=int, default=200,
                        help='how often to write to TB')
    parser.add_argument('--checkpoint_freq', type=int, default=25000,
                        help='how often to save a model ckpt')
    parser.add_argument('--load_from_checkpoint', action='store_true', default=False,
                        help='whether to load training from a checkpoint')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='name of a .pth file')
    parser.add_argument('--run_suffix', type=str, default='debug', 
                        help='string to attach to model name')
    parser.add_argument('--out_dir', type=str, default='experiments',
                        help='output folder for results')
    parser.add_argument('--tqdm', action='store_true', default=False,
                        help='show training progress in CLI')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='run code in debug mode')
    # model
    parser.add_argument('--g_goal', type=float, default=0.5655,
                        help='GECO recon goal')
    parser.add_argument('--g_lr', type=float, default=1e-6, 
                        help='GECO learning rate')
    parser.add_argument('--g_alpha', type=float, default=0.99,
                        help='GECO momentum for error')
    parser.add_argument('--g_init', type=float, default=1.0,
                        help='GECO init Lagrange factor')
    parser.add_argument('--g_min', type=float, default=1e-10,
                        help='GECO min Lagrange factor')
    parser.add_argument('--g_speedup', type=float, default=10.,
                        help='Scale GECO Lr if delta positive')
    ###########################################
    parser.add_argument('--z_dim', type=int, default=64,
                        help='slot dim')
    parser.add_argument('--s_dim', type=int, default=128,
                        help='s dim')
    parser.add_argument('--L', type=int, default=3,
                        help='number of relation layers in scene encoder')
    parser.add_argument('--K', type=int, default=9,
                        help='number of slots')       
    ###########################################                                                             
    parser.add_argument('--kernel', type=str, default='gaussian',
                        help='{laplacian,gaussian,epanechnikov}')
    parser.add_argument('--semiconv', action='store_true', default=True,
                        help='semiconv for GENESISv2')
    parser.add_argument('--dynamic_K', action='store_true', default=False,
                        help='dynamic_K for GENESISv2')
    parser.add_argument('--klm_loss', action='store_true', default=False,
                        help='klm_loss for GENESISv2')
    parser.add_argument('--detach_mr_in_klm', action='store_true', default=True,
                        help='detach_mr_in_klm for GENESISv2')
    parser.add_argument('--pixel_bound',  action='store_true', default=True,
                        help='constrain decoder outputs')
    parser.add_argument('--img_size', type=int, default=64,
                        help='side length of square image inputs')
    parser.add_argument('--pixel_std', type=float, default=0.7,
                        help='Global std dev for image likelihood')
    parser.add_argument('--image_likelihood', type=str, default='MoG-Normed',
                        help='MoG or MoG-Normed (use normalization constant)')

    args = vars(parser.parse_args())

    run(args, args['seed'])