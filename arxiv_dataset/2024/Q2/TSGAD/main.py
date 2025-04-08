from args import create_exp_dirs
from args import init_parser, init_sub_args
import torch
import random
import numpy as np
import lib.dist as dist
from lib.flows import FactorialNormalizingFlow
from dataset import get_dataset_and_loader
from utils.train_utils import dump_args, init_model_params, Trainer, init_optimizer, init_scheduler
from utils.data_utils import trans_list
from models import VAE
from tqdm import tqdm
from utils.eval import score_dataset, get_train_dist
import yaml
import os
from scipy.stats import norm, multivariate_normal

def main ():
    print('sedaye mano darid az Chalotte America!')
    parser = init_parser()
    args = parser.parse_args()
    if args.seed == 999:  # Record and init seed
        args.seed = torch.initial_seed()
        np.random.seed(0)
    else:
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(args.seed)
        np.random.seed(0)
    args, model_args = init_sub_args(args)
    args.ckpt_dir = create_exp_dirs(args.exp_dir, dirmap=args.dataset)
    
    pretrained_model = vars(args).get('model_ckpt_dir', None)
    dataset, loader = get_dataset_and_loader(args, trans_list=trans_list, only_test=(pretrained_model is not None))
    # model_args = init_model_params(args, dataset)
    
    if model_args.model_dist == 'normal':
        prior_dist = dist.Normal()
        q_dist = dist.Normal()
    elif model_args.model_dist == 'laplace':
        prior_dist = dist.Laplace()
        q_dist = dist.Laplace()
    elif model_args.model_dist == 'flow':
        prior_dist = FactorialNormalizingFlow(dim=model_args.model_latent_dim, nsteps=32)
        q_dist = dist.Normal()

    vae = VAE(z_dim=model_args.latent_dim, 
              use_cuda=True, 
              device=args.device,
              prior_dist=prior_dist, 
              q_dist=q_dist,
              include_mutinfo=not args.exclude_mutinfo, 
              tcvae=args.tcvae,  
              conv=args.conv, 
              graph=args.graph,
              mss=args.mss,
              drop_out=args.dropout,
              conv_oper=args.conv_oper,
              act=args.act,
              headless=args.headless,
              input_frames=args.seg_len,
              mse=args.mse,
              alpha=args.model_alpha,
              gamma=args.model_gamma,
              )
    
    if pretrained_model == None:
        if not os.path.exists(args.model_save_dir):
        # Create the directory
            os.makedirs(args.model_save_dir)
        arguments = vars(args)
        with open(args.model_save_dir + '/' + 'arguments.yaml', 'w') as file:
            yaml.dump(arguments, file)
        ae_optimizer_f = init_optimizer(args.model_optimizer, lr=args.model_lr)
        ae_scheduler_f = init_scheduler(args.sched, lr=args.model_lr, epochs=args.epochs)
        trainer = Trainer(model_args, vae, loader['train'], loader['test'], optimizer_f=ae_optimizer_f,
                                scheduler_f=ae_scheduler_f)
        trained_model = trainer.train(checkpoint_filename='vae', args=args)
        
    else:
        checkpoint = torch.load(args.model_ckpt_dir)
        vae.load_state_dict(checkpoint['state_dict'])
        print('Model loaded successfully!')
        vae.to(args.device)
        
    eval_loss = []
    eval_elbo = []
    dataset_size = len(loader['test'].dataset)
    mean, std = get_train_dist (vae, loader['test'], args)
    mean = torch.from_numpy(mean).to(args.device)

    # distribution_m = norm(loc=m_mean, scale=m_std)
    # distribution_v = norm(loc=v_mean, scale=v_std)
    # distribution = multivariate_normal(mean=mean.astype(np.float64), cov=np.diag(std.astype(np.float64)**2))

    vae.eval()
    
    with torch.no_grad():
        for i, data_batch in enumerate(tqdm(loader['test'])):
            data = data_batch[0].to(args.device, non_blocking=True)
            data = data[:,0:2, :, :]
            obj, elbo = vae.elbo(data, dataset_size)
            eval_elbo.extend(elbo.cpu().numpy())
            data = data.view(data.shape[0], 2, args.seg_len, 18)
            _, z_params, _ = vae.encode(data)
            z_params = z_params.view(z_params.shape[0], -1)
            l2_distance = (torch.sqrt(torch.sum((z_params - mean)**2, dim=1))).cpu().numpy()
            eval_loss.extend(l2_distance)
            # probability_m = distribution_m.pdf(z_params[:, :, 0])
            # probability_v = distribution_v.pdf(z_params[:, :, 1])
            # probability = distribution.pdf(z_params.view(data.shape[0], -1).cpu().astype(np.float64))
            
            # Calculate the joint probability by multiplying the probabilities of the two variables
            # joint_probability = probability_m * probability_v
    auc_roc, dp_shift, dp_sigma, auc_pr, eer, eer_th = score_dataset(args.mask_root, np.array(eval_loss), dataset['test'].metadata, save_results=args.save_scores, seg_len=args.seg_len, directory=args.score_save_dir)
    print("*** Normal Dist ***")
    print('AUC ROC: {}'.format(auc_roc))
    print('AUC PR: {}'.format(auc_pr))
    print('EER: {}'.format(eer))
    print('EER TH: {}'.format(eer_th))
    auc_roc, dp_shift, dp_sigma, auc_pr, eer, eer_th = score_dataset(args.mask_root, np.array(eval_elbo), dataset['test'].metadata, save_results=False, seg_len=args.seg_len)

    print("*** ELBO ***")
    print('AUC ROC: {}'.format(auc_roc))
    print('AUC PR: {}'.format(auc_pr))
    print('EER: {}'.format(eer))
    print('EER TH: {}'.format(eer_th))
    
if __name__ == '__main__':
    main()