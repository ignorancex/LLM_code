import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PFINDataset import PFINDataset, JetClassData
from UQPFIN import UQPFIN as Model
import numpy as np
from sklearn.metrics import accuracy_score
from torchinfo import summary
import torch.nn as nn
import glob
import argparse, os, json, sys, random
import torch.distributed as dist

def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def getprobs(outs):
    alphas = outs + 1
    S = torch.sum(alphas, 1).reshape(-1,1)
    return alphas / S

def fgamma(x):
    return torch.exp(torch.lgamma(x))


class LossCE(nn.Module):
    def __init__(self):
        super(LossCE, self).__init__()
        # self.eps=1e-15
    def forward(self, labels, outs):
        # return -(labels * torch.log(outs.clamp(min=eps, max=1-eps))).sum(1).mean()
        return -(labels * torch.log(outs)).sum(1).mean()

    
class LossCE_Bayes(nn.Module):
    def __init__(self):
        super(LossCE_Bayes, self).__init__()
    def forward(self, labels, outs):
        # labels size: (Nb, nclasses) [true values]
        # outs size: (Nb, nclasses) [NN predictions]
        alphas = outs + 1
        S = torch.sum(alphas, 1).reshape(-1,1)
        return (labels * (torch.log(S) - torch.log(alphas))).sum(1).mean()

class LossCE_Gibbs(nn.Module):
    def	__init__(self):
        super(LossCE_Gibbs, self).__init__()
    def forward(self,labels, outs):
        # labels size: (Nb, nclasses) [true values]
        # outs size: (Nb, nclasses) [NN predictions]
        alphas = outs + 1
        S = torch.sum(alphas, 1).reshape(-1,1)
        return (labels * (torch.digamma(S) - torch.digamma(alphas))).sum(1).mean()

class LossMSE(nn.Module):
    def	__init__(self):
        super(LossMSE, self).__init__()
    def forward(self, labels, outs):
        # labels size: (Nb, nclasses) [true values]
        # outs size: (Nb, nclasses) [NN predictions]
        alphas = outs + 1
        S = torch.sum(alphas, 1).reshape(-1,1)
        probs = alphas / S
        return ((labels - probs)**2 + probs * (1 - probs) / (1 + S)).sum(1).mean()

class KLDiv(nn.Module):
    def	__init__(self):
        super(KLDiv, self).__init__()
    def forward(self,labels, outs):
        K = torch.tensor(labels.shape[-1]).float()
        alphas = outs + 1
        _alphas = labels + (1-labels)*alphas
        _S = torch.sum(_alphas, 1).reshape(-1,1)
        lognum = torch.lgamma(_S)
        logden = torch.lgamma(K*1.0) + torch.lgamma(_alphas).sum(1).reshape(-1,1)
        t2 = ((_alphas - 1) * (torch.digamma(_alphas) - torch.digamma(_S) )).sum(1).reshape(-1,1)
        return (lognum - logden + t2).mean()



def save_model(args, epoch, model, optimizer):
    """Saves model checkpoint on given epoch with given data name.
    """
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, args.outdir + '/UQPFIN_best_alldicts' + args.extra_name )
    
    return True

def load_model(rank, epoch, model, optimizer):
    """Loads model state from file to the GPU with given rank.
    """
    torch.distributed.barrier()
    map_location = {'cuda:0': f'cuda:{rank}'}
    checkpoint = torch.load(args.outdir + '/UQPFIN_best_alldicts' + args.extra_name,
                            map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer

def train(gpu, args):

    ## Setting up the process rank
    
    rank = args.local_rank * args.gpus + gpu	                          
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    torch.manual_seed(args.seed)
    device = torch.device('cuda:{}'.format(gpu))
    torch.cuda.set_device(device)


    ## Setting up model parameters

    if args.data_type == 'topdata':
        features = 3
        Np = 60
        num_classes = 2
        skiplabels = []
        label_indices = [0,1]
    elif args.data_type == 'jetnet':
        features = 3    
        Np = 30
        if args.skiplabels:
            skiplabels = list(map(int, args.skiplabels.strip().split(',')))
        else:
            skiplabels = []
        num_classes = 5 - len(skiplabels)
        label_indices = [i for i in range(5) if i not in skiplabels]
    elif args.data_type == 'jetclass':
        features = 11
        Np = 60
        if args.skiplabels:
            skiplabels = list(map(int, args.skiplabels.strip().split(',')))
        else:
            skiplabels = []
        num_classes = 10 - len(skiplabels)
        label_indices = [i for i in range(10) if i not in skiplabels]
        
    ## defining the model
    model = Model(particle_feats = features,
                  n_consts = Np,
                  num_classes = num_classes,
                  device = device,
                  PhiI_nodes = args.n_phiI,
                  interaction_mode = args.x_mode,
                  use_softmax = bool(args.use_softmax),
                  use_dropout = bool(args.use_dropout),
                  Phi_sizes = list(map(int, args.phi_nodes.split(','))),
                  F_sizes   = list(map(int, args.f_nodes.split(',')))).to(device)
    if rank == 0:
        summary(model, ((1, Np, features), (1,7), (1, 1, Np)))

    ## preload model
    if args.preload and os.path.exists(args.preload_file):
        model_checkpoint = model.state_dict()
        for key in model_checkpoint.keys():
            model_checkpoint[key] *= 0. 
        old_checkpoint = torch.load(args.preload_file, map_location = device)
        for key in old_checkpoint:
            if key in model_checkpoint.keys():
                if 'weight' in key:
                    d0 = min(model_checkpoint[key].shape[0], old_checkpoint[key].shape[0])
                    d1 = min(model_checkpoint[key].shape[1], old_checkpoint[key].shape[1])
                    model_checkpoint[key][:d0, :d1] = old_checkpoint[key][:d0, :d1]
                if 'bias' in key:
                    d0 = min(model_checkpoint[key].shape[0], old_checkpoint[key].shape[0])
                    model_checkpoint[key][:d0] = old_checkpoint[key][:d0]
        model.load_state_dict(model_checkpoint)

    ## creating model instance with DDP
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    ## loading data
    if args.data_type in ['topdata', 'jetnet']:
        train_path = args.data_loc + '/' + args.data_type + '/processed/train.h5'
        val_path   = args.data_loc + '/' + args.data_type + '/processed/val.h5'
        train_set = PFINDataset(train_path)
        val_set = PFINDataset(val_path)
        sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        trainloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, 
                                 num_workers=1*args.gpus, pin_memory=True, sampler = sampler)
        # validation is only done on rank = 0
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, 
                                num_workers=4, pin_memory=True, persistent_workers=True)
    else:
        assert args.ndata != 0, "--ndata should not be 0"
        train_DS = JetClassData(batch_size = args.batch_size)
        val_DS = JetClassData(batch_size = args.batch_size)
        all_train_files = sorted(glob.glob(os.path.join(args.data_loc + '/' + args.data_type, "/processed/train_*.h5")))[0:args.ndata_per_gpu*args.world_size]
        all_val_files = sorted(glob.glob(os.path.join(args.data_loc + '/' + args.data_type, "/processed/val_*.h5")))[0:2]
        Nfiles_per_gpu = len(all_train_files) // args.world_size
        train_DS.set_file_names(file_names = all_train_files[rank*Nfiles_per_gpu:(rank+1)*Nfiles_per_gpu])
        val_DS.set_file_names(file_names = all_val_files)
        if rank == 0:
            print('Number of training Files:', len(all_train_files))
            print('Training Files:', all_train_files)
        print("Indices selected for rank {}: {} - {}".format(rank, rank*Nfiles_per_gpu, (rank+1)*Nfiles_per_gpu))



    ## defining optimizer parameters
    
    l_rate = 1e-3
    opt_weight_decay = 0
    opt = torch.optim.Adam(model.parameters(),  lr=l_rate, weight_decay=opt_weight_decay)

    ## getting index filtering logics
    
    m_logic, (m1, m2) = args.massrange.strip().split(':')[0], list(map(float, args.massrange.strip().split(':')[1].split(',')))
    pt_logic, (pt1, pt2) = args.ptrange.strip().split(':')[0], list(map(float, args.ptrange.strip().split(':')[1].split(',')))
    eta_logic, (eta1, eta2) = args.etarange.strip().split(':')[0], list(map(float, args.etarange.strip().split(':')[1].split(',')))

    if rank == 0:
        print("data type: ", args.data_type)
        print("m_logic, pt_logic, eta_logic: ", m_logic, pt_logic, eta_logic)
        print("m1, m2, pt1, pt2, eta1, eta2: ", m1, m2, pt1, pt2, eta1, eta2)
        print("skip labels: ", skiplabels)
        print("classes: ", num_classes)

    ## definging the loss functions
    
    kldiv = KLDiv()
    if args.use_softmax:
        crit = LossCE()
    else:
        crit = LossMSE()
    crit.cuda(rank)
    kldiv.cuda(rank)

    ## starting training
    best_val_acc = 0
    no_change = 0
    pre_val_acc = 0
    epoch = 0
    restart_count = 0
    epochs = args.epochs
    print("Training starting on GPU: ", rank)
    while epoch < epochs:
        if args.data_type == 'jetclass':
            trainloader = train_DS.generate_data()
            val_loader = val_DS.generate_data()
        l = min(1.0, epoch/10.)
        if "nominal" in args.klcoef:
            if 'slope' in args.klcoef:
                l = min(float(args.klcoef.split('_')[1]), epoch * float(args.klcoef.split('_')[3]))
            elif '_' in args.klcoef:
                l = min(float(args.klcoef.split('_')[1]), epoch/10.)
            else:
                l = min(1.0, epoch/10.)
        if rank == 0 and "nominal" in args.klcoef:
            print("L = {}".format(l))
        val_loss_total = 0
        train_loss_total = 0
        train_acc_total = 0
        val_acc_total = 0

        #train loop

        model.train()
        ntrain = 0
        for x,m,a,y in tqdm(trainloader, disable=args.batchmode):
            if m_logic == 'AND':
                keep_masses = (a[:, 1] > min(m1,m2)) & (a[:, 1] < max(m1,m2))
            else:
                keep_masses = (a[:, 1] < min(m1,m2)) | (a[:, 1] > max(m1,m2))

            if pt_logic == 'AND':
                keep_pts = (a[:, 2] > min(pt1,pt2)) & (a[:, 2] < max(pt1,pt2))
            else:
                keep_pts = (a[:, 2] < min(pt1,pt2)) | (a[:, 2] > max(pt1,pt2))

            if eta_logic == 'AND':
                keep_etas = (a[:, 3] > min(eta1,eta2)) & (a[:, 3] < max(eta1,eta2))
            else:
                keep_etas = (a[:, 3] < min(eta1,eta2)) | (a[:, 3] > max(eta1,eta2))
            
            keep_labels = torch.tensor(np.isin(np.argmax(y.cpu().numpy(), 1), skiplabels, invert=True)).bool()
            
            keep = (keep_masses & keep_pts & keep_etas & keep_labels).bool()
            ntrain += keep.sum().item()
            
            opt.zero_grad()
            x = x[keep].to(device)
            m = m[keep].to(device)
            y = y[keep].to(device)
            y = y[:, label_indices]
            if y.sum().int().item() != keep.sum().int().item():
                print("The class probabilities don't add up to 1, please check! Numer of events: {}, Sum of probs: {}".format(keep.sum().int().item(), y.sum().int().item()))
                # sys.exit(1)
                
            a = a[keep].to(device)
            pred = model(x,a,m)

            if args.use_softmax:
                loss = crit(y, pred)
            elif args.klcoef == "0" or l == 0.:
                loss = crit(y, pred)
            elif args.klcoef == "nominal":
                loss = crit(y, pred) + l * kldiv(y, pred)
            else:
                loss = crit(y, pred) + float(args.klcoef)*kldiv(y, pred)

            train_loss_total += loss.item() * keep.sum().float().item()

            with torch.no_grad():
                if args.use_softmax:
                    probs = pred
                else:
                    probs = getprobs(pred)
                acc = accuracy_score(  np.argmax(probs.cpu().numpy(), 1),
                                       np.argmax(y.cpu().numpy(), 1),
                                       normalize = False )
                train_acc_total += acc

            loss.backward()
            opt.step()

        # Validation loop
        if rank == 0:
            model.eval()
            nval = 0
            
            with torch.no_grad():
                for x,m,a,y in tqdm(val_loader, disable=args.batchmode):
                
                    if m_logic == 'AND':
                        keep_masses = (a[:, 1] > min(m1,m2)) & (a[:, 1] < max(m1,m2))
                    else:
                        keep_masses = (a[:, 1] < min(m1,m2)) | (a[:, 1] > max(m1,m2))


                    if pt_logic == 'AND':
                        keep_pts = (a[:, 2] > min(pt1,pt2)) & (a[:, 2] < max(pt1,pt2))
                    else:
                        keep_pts = (a[:, 2] < min(pt1,pt2)) | (a[:, 2] > max(pt1,pt2))


                    if eta_logic == 'AND':
                        keep_etas = (a[:, 3] > min(eta1,eta2)) & (a[:, 3] < max(eta1,eta2))
                    else:
                        keep_etas = (a[:, 3] < min(eta1,eta2)) | (a[:, 3] > max(eta1,eta2))

                    
                    keep_labels = torch.tensor(np.isin(np.argmax(y.cpu().numpy(), 1), skiplabels, invert=True)).bool()
                    keep = (keep_masses & keep_pts & keep_etas & keep_labels).bool()
                    nval += keep.sum().item()
                    
                    x = x[keep].to(device)
                    m = m[keep].to(device)
                    y = y[keep].to(device)
                    y = y[:, label_indices]
                    a = a[keep].to(device)
                    if y.sum().int().item() != keep.sum().int().item():
                        print("The class probabilities don't add up to 1, please check! Numer of events: {}, Sum of probs: {}".format(keep.sum().int().item(), y.sum().int().item()))
                        # sys.exit(1)
                    pred = model(x,a,m)

                    if args.use_softmax:
                        loss = crit(y, pred)
                    elif args.klcoef == "nominal":
                        loss = crit(y, pred) + l*kldiv(y, pred)
                    elif args.klcoef == "0":
                        loss = crit(y, pred)
                    else:
                        loss = crit(y, pred) + float(args.klcoef)*kldiv(y,pred)

                    val_loss_total += loss.item() * keep.sum().float().item()
                    if args.use_softmax:
                        probs = pred
                    else:
                        probs = getprobs(pred)
                    acc =   accuracy_score( np.argmax(probs.cpu().numpy(), 1),
                                            np.argmax(y.cpu().numpy(), 1),
                                            normalize = False )
                    val_acc_total += acc
            train_loss_total /= ntrain #len(train_set)
            val_loss_total /= nval #len(val_set)
            val_acc_total /= nval #len(val_set)
            train_acc_total /= ntrain #len(train_set)
            print('Epoch: ', epoch)
            print('Best Validation Accuracy: ' + str(best_val_acc))
            print('Current Validation Accuracy: ' + str(val_acc_total))
            print('Current Training Accuracy: ' + str(train_acc_total))
            print('Current Validation Loss: ' + str(val_loss_total))
            print('Current Training Loss: ' + str(train_loss_total))
                

            if val_acc_total > best_val_acc:
                print('Saving best model based on accuracy')
                torch.save(model.state_dict(), args.outdir + '/UQPFIN_best'+ args.extra_name)
                save_model(args, epoch, model, opt)
                best_val_acc = val_acc_total

            # if epoch > 2 and best_val_acc - val_acc_total > 0.1:
            #     print('Validation accuracy dropped by more than 10%. Reloading best model')
            #     model, opt = load_model(rank, epoch, model, opt)
            #     # model.load_state_dict(torch.load(args.outdir + '/UQPFIN_best'+extra_name, map_location=device))

            # if epoch > 2 and best_val_acc < 0.55:
            #     print('Validation accuracy is close to 0.5. Resetting the model')
            #     del model, opt
            #     model = Model(particle_feats = features,
            #                   n_consts = Np,
            #                   num_classes = num_classes,
            #                   device = device,
            #                   PhiI_nodes = args.n_phiI,
            #                   interaction_mode = args.x_mode,
            #                   Phi_sizes = list(map(int, args.phi_nodes.split(','))),
            #                   F_sizes   = list(map(int, args.f_nodes.split(',')))).to(device)
            #     opt = torch.optim.Adam(model.parameters(),  lr=l_rate, weight_decay=opt_weight_decay)
            #     restart_count += 1
            #     epoch = 0
            #     best_val_acc = 0
            #     no_change = 0
            #     pre_val_acc = 0
            #     if restart_count <= 3:
            #         continue
            #     else:
            #         print("Model did not improve after 3 restarts! Check!")
            #         sys.exit(1)

        pre_val_acc = val_acc_total
        epoch += 1

    print("Training complete for GPU: ", rank)
    if rank == 0:
        print('Saving last model')
        torch.save(model.state_dict(), args.outdir + '/UQPFIN_last'+args.extra_name)
    torch.distributed.destroy_process_group()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--outdir", type=str, action="store", dest="outdir", default="./trained_models/", help="Output directory for trained model" )
    parser.add_argument("--outdictdir", type=str, action="store", dest="outdictdir", default="./trained_model_dicts/", help="Output directory for trained model metadata" )
    parser.add_argument("--Np", type=int, action="store", dest="Np", default=60, help="Number of constituents") # cannot be changed with current dataloader
    parser.add_argument("--NPhiI", type=int, action="store", dest="n_phiI", default=128, help="Number of hidden layer nodes for Interaction Phi")
    parser.add_argument("--x-mode", type=str, action="store", dest="x_mode", default="sum", help="Mode of Interaction: ['sum', 'cat']")
    parser.add_argument("--Phi-nodes", type=str, action="store", dest="phi_nodes", default="100,100,64", help="Comma-separated list of hidden layer nodes for Phi")
    parser.add_argument("--F-nodes", type=str, action="store", dest="f_nodes", default="64,100,100", help="Comma-separated list of hidden layer nodes for F")
    parser.add_argument("--epochs", type=int, action="store", dest="epochs", default=50, help="Epochs")
    parser.add_argument("--label", type=str, action="store", dest="label", default="", help="a label for the model")
    parser.add_argument("--batch-size", type=int, action="store", dest="batch_size", default=250, help="batch_size")
    parser.add_argument("--data-loc", type=str, action="store", dest="data_loc", default="../datasets/", help="Directory for data" )
    parser.add_argument("--data-type", type=str, action="store", dest="data_type", default="topdata", help="Dataset to train on" )
    parser.add_argument("--preload", action="store_true", dest="preload", default=False, help="Preload weights and biases from a pre-trained Model")
    parser.add_argument("--preload-file", type=str, action="store", dest="preload_file", default="", help="Location of the model to the preload" )
    parser.add_argument("--KLcoef", type=str, action="store", dest="klcoef", default="0", help="annealing coefficient, nominal implies an epoch-dependent coefficient")
    parser.add_argument("--mass-range", type=str, action="store", dest="massrange", default='AND:0,10000.',
                        help="thresholds for jet mass range in the form 'LOGIC:M1,M2' where LOGIC can be AND (between the range) or OR (between the range)")
    parser.add_argument("--pt-range", type=str, action="store", dest="ptrange", default='AND:0,10000',
                        help="thresholds for jet pT range in the form 'LOGIC:PT1,PT2' where LOGIC can be AND (between the range) or OR (outside the range)")
    parser.add_argument("--eta-range", type=str, action="store", dest="etarange", default='AND:-6,6',
                        help="thresholds for jet eta range in the form 'LOGIC:ETA1,ETA2' where LOGIC can be AND (between the range) or OR (outside the range)")
    parser.add_argument("--skip-labels", type=str, action="store", dest="skiplabels", default='',
                        help="Jet labels to drop from training")
    parser.add_argument("--batch-mode", action="store_true", dest="batchmode", default=False, help="Set this flag when running in batch mode to suppress tqdm progress bars")
    parser.add_argument("--use-softmax", action="store_true", dest="use_softmax", default=False, help="Set this flag when using softmax probabilites")
    parser.add_argument("--use-dropout", action="store_true", dest="use_dropout", default=False, help="Set this flag when using dropout layers")
    parser.add_argument('--load-json', type=str, action="store", dest="load_json", default="", help='Load settings from file in json format. Command line options override values in file.')
    parser.add_argument('--ndata-per-gpu', default=1, type=int, help='Only for jetclass data- number of data files per gpu (1 file = 1M jets')
    parser.add_argument('--nodes', default=1, type=int, help='Number of nodes (do not change it yet)')
    parser.add_argument('--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('--local-rank', default=0, type=int, help='ranking within the nodes')
    parser.add_argument('--IP', default='localhost', type=str, help='IP Address of Master Node')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    
    
    args = parser.parse_args()
    
    if args.load_json:
        with open(args.load_json, 'rt') as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            args = parser.parse_args(namespace=t_args)
    
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = args.IP
    os.environ['MASTER_PORT'] = '8888'
    
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
    if not os.path.exists(args.outdictdir):
        os.mkdir(args.outdictdir)
        
    extra_name = args.label
    if extra_name != "" and not extra_name.startswith('_'):
        extra_name = '_' + extra_name
    
    args.extra_name = extra_name
    model_dict = {}
    for arg in vars(args):
        if arg == 'load_json':
            continue
        model_dict[arg] = getattr(args, arg)
    f_model = open("{}/UQPFIN{}.json".format(args.outdictdir, extra_name), "w")
    json.dump(model_dict, f_model, indent=3)
    f_model.close()

    torch.multiprocessing.spawn(train, args=(args, ), nprocs=args.world_size, join=True)


