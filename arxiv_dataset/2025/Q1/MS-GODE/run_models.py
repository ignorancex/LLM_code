import os
import pickle
from datetime import datetime
import time

from lib.new_dataLoader import ParseData
import argparse
import traceback
import numpy as np
import torch
import torch.optim as optim
from torch.distributions.normal import Normal
from lib.gnn_models import set_model_task, get_model_scores
from distutils.util import strtobool
from lib.utils import strtobool_str
from cl_baselines import *
from lib.utils import read_selected_VCell_vars
from lib.create_latent_ode_model import create_LatentGODE_model, create_LatentODE_mask_model, create_LatentODE_model,create_CoupledODE_model

# Generative model for noisy data based on ODE
parser = argparse.ArgumentParser('Latent ODE')
parser.add_argument('--n-balls', type=int, default=None, help='Number of objects in the dataset.')
parser.add_argument('--nepos', type=int, default=2) # 20 performs much better than 50, according to visualization results
parser.add_argument('--lr',  type=float, default=5e-4, help="Starting learning rate.default=5e-4")
parser.add_argument('-b', '--batch-size', type=int, default=1)
parser.add_argument('--cut_num',type=int, default=2, help='load partial data for testing the code, specify as None to avoid mannual cut_num')
parser.add_argument('--trainset_size',type=int, default=20000, help='load partial data for testing the code, specify as None to avoid mannual cut_num')
parser.add_argument('--testset_size',type=int, default=5000, help='load partial data for testing the code, specify as None to avoid mannual cut_num')
parser.add_argument('--sampled_data', type=strtobool_str, default=False, help='whether use the data subsampled during generation')
parser.add_argument('--normalizeVCellfeat', type=str, default='universal', help='normalization mode, individual or universal')
parser.add_argument('--normalizeVCelltime', type=strtobool_str, default=True, help='whether use the data subsampled during generation')
parser.add_argument('--VCell_window_size', type=float, default=10, help='window size for temporal aggregation')
parser.add_argument('--offset_extrap', type=strtobool_str, default=True, help='whether make extrap time start at 0, current for VCell')
parser.add_argument('--keep_original_edge_types', type=strtobool_str, default=False, help='whether make extrap time start at 0, current for VCell')

parser.add_argument('--save', type=str, default='experiments/', help="Path for save checkpoints")
parser.add_argument('--save-graph', type=str, default='plot/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None, help="name of ckpt. If None, run a new experiment.")
parser.add_argument('--fix_random_seed', type=strtobool_str, default=False, help="whether to use Random_seed")
parser.add_argument('-r', '--random-seed', type=int, default=1991, help="Random_seed")
parser.add_argument('--z0-encoder', type=str, default='GTrans', help="GTrans")
parser.add_argument('-l', '--latents', type=int, default=16, help="Size of the latent state")
parser.add_argument('--rec-dims', type=int, default= 64, help="Dimensionality of the recognition model .")
parser.add_argument('--ode-dims', type=int, default=128, help="Dimensionality of the ODE func")
parser.add_argument('--rec-layers', type=int, default=2, help="Number of layers in recognition model ")
parser.add_argument('--n-heads', type=int, default=1, help="Number of heads in GTrans")
parser.add_argument('--gen-layers', type=int, default=1, help="Number of layers  ODE func ")
parser.add_argument('--mode', type=str,default="ex", help="[ex,in],Set extrapolation/interpolation mode. ")
parser.add_argument('--dropout', type=float, default=0.2,help='Dropout rate (1 - keep probability).')
parser.add_argument('--dropout_mask', type=float, default=0.0,help='Dropout rate when using masks.')
parser.add_argument('--sample-percent-train', type=float, default=0.6,help='Percentage of training observtaion data preserved during data loading')
parser.add_argument('--sample-percent-test', type=float, default=0.6,help='Percentage of testing observtaion data preserved during data loading')
parser.add_argument('--extrap-percent-train', type=float, default=0.6,help='Percentage of observtaion data length when training, currently for VCell')
parser.add_argument('--extrap-percent-test', type=float, default=0.6,help='Percentage of observtaion data length when testing, currently for VCell')
parser.add_argument('--augment_dim', type=int, default=64, help='augmented dimension')
parser.add_argument('--edge_types', type=int, default=2, help='edge number in NRI')
parser.add_argument('--odenet', type=str, default="NRI", help='NRI')
parser.add_argument('--solver', type=str, default="rk4", help='dopri5,rk4,euler')
parser.add_argument('--l2', type=float, default=1e-3, help='l2 regulazer')
parser.add_argument('--optimizer', type=str, default="AdamW", help='Adam, AdamW')
parser.add_argument('--clip', type=float, default=10, help='Gradient Norm Clipping')
parser.add_argument('--cutting_edge', type=strtobool, default=True, help='True/False')
parser.add_argument('--extrap_num', type=int, default=40, help='extrap num ')
parser.add_argument('--rec_attention', type=str, default="attention")
parser.add_argument('--alias', type=str, default="run")

parser.add_argument('--thresholding', type=str, default='fast', help='[topk, fast]whether to select top k% scores or positive scores')
parser.add_argument('--mask',type=strtobool, default=True, help='whether to use mask on the weights, for continual learning')
parser.add_argument('--joint_skip',type=strtobool, default=False, help='whether to only train at the end or train at each task')
parser.add_argument('--data', type=str, default='None', help="springs,charged,motion")
parser.add_argument('--system', type=str, default='VCell', help="physics,motion,NRW,VCell")
parser.add_argument('--tasks',default=None)

parser.add_argument('--baseline', type=str,default='er',help='bare,ewc,mas,lwf,gem,er,joint,msode,bcsr,biascorr,sdp') # sdp: scheduled data prior
parser.add_argument('--backbone', type=str,default='cgode',help='LGODE, LODE,cgode') # without mask, what backbone is used
parser.add_argument('--ewc_args', default={'memory_strength':10000.})
parser.add_argument('--mas_args', default={'memory_strength':10000.})
parser.add_argument('--mas_args_memory_strength', type=float, default=10000.)
parser.add_argument('--lwf_args', default={'lambda_dist': 1.})
parser.add_argument('--er_args', default={'budget': 8})
parser.add_argument('--er_args_memory_budget', type=int, default=8)
parser.add_argument('--er_args_buffered_ids', type=int, default=8)
parser.add_argument('--gem_args', default={'budget': 8})
parser.add_argument('--bcsr_args', default={'budget': 8})
parser.add_argument('--joint_args', default=None)

parser.add_argument('--device',type=int, default=0)
parser.add_argument('--save_results',type=strtobool,default=True)
parser.add_argument('--n_iters_to_viz',type=int,default=10)
parser.add_argument('--repeats',type=int,default=1)
parser.add_argument('--mask_criteria',type=str,default='loss',help='use likelihood (loss) or mse to choose mask')
parser.add_argument('--motion_fully_connect', type=strtobool, default=False)
parser.add_argument('--save_predicted_trajs', type=strtobool, default=False)
parser.add_argument('--overwrite_results', type=strtobool, default=True)
parser.add_argument('--mask_select', type=str, default='part_recon', choices=['one_shot_gradient','weighted_sum','mask_prototype', 'part_recon']) #  one_shot_gradient, weighted_sum, mask_prototype

# arguments for LODE
parser.add_argument('--poisson', type=strtobool, default=True, help='argument for LODE')
parser.add_argument('-u', '--units', type=int, default=100, help="Number of units per layer in ODE func")
parser.add_argument('--z0-encoder-lode', type=str, default='odernn', help="Type of encoder for Latent ODE model: odernn or rnn")
parser.add_argument('-g', '--gru-units', type=int, default=100, help="Number of units per layer in each of GRU update networks")
parser.add_argument('--classif', action='store_true', help="Include binary classification loss -- used for Physionet dataset for hospiral mortality")
parser.add_argument('--linear-classif', action='store_true', help="If using a classifier, use a linear classifier instead of 1-layer NN")

args = parser.parse_args()
assert(int(args.rec_dims%args.n_heads) ==0)
args.mas_args['memory_strength']=args.mas_args_memory_strength
args.er_args['budget']=args.er_args_memory_budget

if args.mask:
    args.dropout=args.dropout_mask
    args.baseline='bare'
if args.tasks is None:
    tasks = [['s',5,5.0,0.1],['c',3,10.0,1.0],['s',5,3.0,0.01]] # 0,1,2
    tasks_VCell = [['egfr','00','00',0.4,1],['Ran','00','00',1.0,1],['egfr','03','00',0.4,1],['Ran','01','00',1.0,1]] #  model_name, vars_config, config, args.sample_percent_train, n_hops
    if args.system == 'physics':
        args.tasks = tasks
    elif args.system == 'VCell':
        args.tasks = tasks_VCell
else:
    args.tasks = utils.str2tasks(args.tasks,system=args.system)

args.er_args_buffered_ids = [random.choices(list(range(args.cut_num)),k=args.er_args['budget']) for i in range(len(args.tasks))]

print(f'tasks are {args.tasks}')
ts = ''
for t in args.tasks:
    if args.system in ['physics']:
        ts += f'{t[0]}_{str(t[1])}_{str(t[2])}_{str(t[3])}_'
    elif args.system=='VCell':
        ts += f'{t[0]}_{t[1]}_{t[2]}_{t[3]}_{t[4]}'
cl_method = cl_dict[args.baseline](args)

#####################################################################################################

def main(args,model=None, task_id=None, train_model = True, motion=False, mask_id=None):
    # prepare tasks
    t0 = time.time()

    data_dict = {'s':'springs','c':'charged','springs':'springs','charged':'charged'}
    if args.system == 'physics':
        args.dataset = f'data/simulated/train{args.trainset_size}_test{args.testset_size}' #'data/example_data'
        args.data, args.n_balls, box_size, interaction_strength = args.tasks[task_id]
        args.suffix = f"_{data_dict[args.data]}{args.n_balls}_{box_size}_{interaction_strength}"
        args.suffix_all = [f"_{data_dict[task[0]]}{task[1]}_{task[2]}_{task[3]}" for task in args.tasks]    #joint和er跑错了，因为task[0]原来用的是args.data，ts又标记的是对的，和数据对不上了
        args.total_ode_step = 60

    elif args.system=='VCell':
        args.dataset = f'data/simulated/VCell'  # 'data/example_data'
        model_name, vars_config, config, args.sample_percent_train, n_hops = args.tasks[task_id][0], args.tasks[task_id][1], args.tasks[task_id][2], float(args.tasks[task_id][3]), int(args.tasks[task_id][4])
        vars_config_path = f'./data/{model_name}_selected_vars_{vars_config}.txt'
        args.n_balls = len(read_selected_VCell_vars(vars_config_path))
        args.sample_percent_test = args.sample_percent_train
        args.suffix = f"VCell{model_name}_config{config}_Vars{vars_config}_sampling_{args.sampled_data}_{n_hops}hop"
        args.suffix_all = [f"VCell{task[0]}_config{task[2]}_Vars{task[1]}_sampling_{args.sampled_data}_{n_hops}hop" for task in args.tasks]
        args.total_ode_step = 0 # should not be used for VCell
    else:
        print('Invalid data name')
        exit()

    ############ CPU AND GPU related, Mode related, Dataset Related
    if torch.cuda.is_available():
        gpustr="Using GPU" + "-" * 10
        device = torch.device(f"cuda:{args.device}")
    else:
        gpustr="Using CPU" + "-" * 10
        device = torch.device("cpu")
    if task_id==0 and train_model:
        print(f"Running {args.mode} mode" + "-" * 10 + gpustr)
    if args.fix_random_seed:
        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)
    print('torch seed is ',torch.seed())


    ############ Saving Path and Preload.
    utils.makedirs(args.save)
    utils.makedirs(args.save_graph)

    experimentID = datetime.now().strftime("%d%m%Y_%H%M%S") # args.load

    ############ Loading Data

    if task_id == 0 and train_model:
        print("Loading dataset: " + args.dataset)
    dataloader = ParseData(args.dataset,suffix=args.suffix,mode=args.mode, args =args)
    data_load_func = dataloader.load_data_joint if args.baseline in ['er','joint','biascorr'] else dataloader.load_data
    if args.backbone in ['lode','LODE']:
        data_load_func = dataloader.load_data_lode
    ccn = args.er_args['budget']//len(args.tasks) if args.baseline=='er' else args.cut_num
    test_encoder, test_decoder,  test_graph, test_batch = data_load_func(args,sample_percent=args.sample_percent_test, batch_size=args.batch_size,data_type="test", customize_cut_num=args.cut_num)
    test_batch = max(test_batch) if args.baseline in ['joint','er','biascorr'] else test_batch
    train_encoder,train_decoder, train_graph,train_batch = data_load_func(args,sample_percent=args.sample_percent_train,batch_size=args.batch_size,data_type="train", customize_cut_num = ccn)
    train_batch = max(train_batch) if args.baseline in ['joint','er','biascorr'] else train_batch
    test_enc_mask, test_dec_mask, test_g_mask, test_b_mask = data_load_func(args,sample_percent=args.sample_percent_test, batch_size=args.batch_size, data_type="test",
        customize_cut_num=args.cut_num,mask_select=True)

    args.bcsr_load_func = dataloader.load_data_single_task
    input_dim = dataloader.feature

    ############ Model Select
    # Create the model
    obsrv_std = torch.Tensor([0.01]).to(device)
    z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))

    if model==None:
        if not args.mask:
            model_dict = {'LGODE':create_LatentGODE_model, 'LODE':create_LatentODE_model, 'cgode':create_CoupledODE_model}
            model_create_func = model_dict[args.backbone]
            model = model_create_func(args, input_dim, z0_prior, obsrv_std, device)
            if args.baseline == 'bcsr':
                cl_method.initialize_proxy(model_create_func(args, input_dim, z0_prior, obsrv_std, device))
        elif args.mask:
            model = create_LatentODE_mask_model(args, input_dim, z0_prior, obsrv_std, device)
            set_model_task(model, task_id, verbose=False)
            model.num_tasks = len(args.tasks)
    else:
        # if model is given
        if train_model and args.mask:
            # manually select mask when training
            set_model_task(model, task_id, verbose=False)
    ##################################################################
    # Optimizer
    optimizers = {"AdamW":optim.AdamW,"Adam":optim.Adam}
    optimizer =optimizers[args.optimizer](model.parameters(),lr=args.lr,weight_decay=args.l2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000, eta_min=1e-9)

    wait_until_kl_inc = 10
    best_test_mse = np.inf

    def train_single_batch(model,batch_dict_encoder,batch_dict_decoder,batch_dict_graph,kl_coef,epo):
        train_res = cl_method.observe(args, model, task_id, optimizer, batch_dict_encoder, batch_dict_decoder, batch_dict_graph, n_traj_samples=3, kl_coef=kl_coef,dataloader=dataloader, n_epoch=epo)
        loss = train_res["loss"]
        loss_value = loss.data.item()
        del loss
        torch.cuda.empty_cache()
        # train_res, loss
        return loss_value,train_res["mse"],train_res["likelihood"],train_res["kl_first_p"],train_res["std_first_p"]

    def train_epoch(epo):
        model.train()
        loss_list,mse_list,likelihood_list,kl_first_p_list,std_first_p_list = [],[],[],[],[]

        torch.cuda.empty_cache()

        wait_until_kl_inc = 10

        for itr in range(train_batch):
            kl_coef = 0. if itr < wait_until_kl_inc else (1 - 0.99 ** (itr - wait_until_kl_inc))

            if args.baseline in ['joint','er','biascorr']:
                batch_dict_encoder = [utils.get_next_batch_new(t, device) for t in train_encoder]
                batch_dict_graph = [utils.get_next_batch_new(t, device) for t in train_graph]
                batch_dict_decoder = [utils.get_next_batch(t, device) for t in train_decoder]
            else:
                batch_dict_encoder = utils.get_next_batch_new(train_encoder, device)
                batch_dict_graph = utils.get_next_batch_new(train_graph, device)
                batch_dict_decoder = utils.get_next_batch(train_decoder, device)

            loss, mse,likelihood,kl_first_p,std_first_p = train_single_batch(model,batch_dict_encoder,batch_dict_decoder,batch_dict_graph,kl_coef,epo)

            #saving results
            loss_list.append(loss), mse_list.append(mse), likelihood_list.append(likelihood)
            kl_first_p_list.append(kl_first_p), std_first_p_list.append(std_first_p)

            del batch_dict_encoder, batch_dict_graph, batch_dict_decoder
            torch.cuda.empty_cache()

        scheduler.step()

        message_train = 'Epoch {:04d} [Train seq (cond on sampled tp)] | Loss {:.6f} | MSE {:.6F} | Likelihood {:.6f} | KL fp {:.4f} | FP STD {:.4f}|'.format(
            epo,
            np.mean(loss_list), np.mean(mse_list), np.mean(likelihood_list),
            np.mean(kl_first_p_list), np.mean(std_first_p_list))

        return message_train,kl_coef,np.mean(mse_list)

    train_epoch_end = args.nepos + 1 if train_model else 1
    train_epoch_start = 1 if train_model else 0

    start_mse,end_mse = [],[]



    for epo in range(train_epoch_start, train_epoch_end):

        if train_model:
            message_train, kl_coef, mse = train_epoch(epo)


        else:
            message_train, kl_coef, mse = 'Epoch {:04d} [Train seq (cond on sampled tp)] | Loss {:.6f} | MSE {:.6F} | Likelihood {:.6f} | KL fp {:.4f} | FP STD {:.4f}|'.format(
            0,0,0,0,0,0), 0., 0.

        if epo % args.n_iters_to_viz == 0 or epo in [1,args.nepos]:
            # test model
            model.eval()

            test_encoder_= test_encoder[task_id] if args.baseline  in ['joint','er'] else test_encoder
            test_graph_ = test_graph[task_id] if args.baseline  in ['joint','er'] else test_graph
            test_decoder_ = test_decoder[task_id] if args.baseline  in ['joint','er'] else test_decoder

            test_res = compute_loss_all_batches(args,model, test_encoder_, test_graph_, test_decoder_, test_enc_mask, test_g_mask, test_dec_mask,
                                                n_batches=test_batch, device=device,
                                                n_traj_samples=3, kl_coef=kl_coef, task_id=mask_id)

            message_test = 'Epoch {:04d} [Test seq (cond on sampled tp)] | Loss {:.6f} | MSE {:.6F} | Likelihood {:.6f} | KL fp {:.4f} | FP STD {:.4f}|'.format(
                epo,
                test_res["loss"], test_res["mse"], test_res["likelihood"],
                test_res["kl_first_p"], test_res["std_first_p"])
            print("data: %s, encoder: %s, sample: %s, mode:%s" % (
                args.data, args.z0_encoder, str(args.sample_percent_train), args.mode))
            print(message_train)
            print(message_test)
            print("KL coef: {}".format(kl_coef))

            # record starting and ending performance
            if epo==1:
                start_mse.append(mse)
                start_mse.append(test_res["mse"])
            if epo==args.nepos:
                end_mse.append(mse)
                end_mse.append(test_res["mse"])

            if test_res["mse"] < best_test_mse:
                best_test_mse = test_res["mse"]
                ckpt_path = os.path.join(args.save, "experiment_" + str(
                    experimentID) + "_" + args.z0_encoder + "_" + args.data + "_" + str(
                    args.sample_percent_train) + "_" + args.mode + "_epoch_" + str(epo) + "_mse_" + str(
                    best_test_mse) + '.ckpt')

                #torch.save({'args': args,'state_dict': model.state_dict(),}, ckpt_path)

            torch.cuda.empty_cache()


    return model,best_test_mse,start_mse,end_mse


if __name__ == '__main__':
    name = f"{args.baseline}_{ts}_{args.sample_percent_train}_{args.sample_percent_test}_{args.mode}_mask{args.mask}_mode{args.mask_select}_dropMASK{args.dropout_mask}_ts{args.thresholding}_{args.latents}_{args.rec_dims}_{args.ode_dims}_e{args.nepos}_b{args.batch_size}_lr{args.lr}_ite{args.repeats}_cutnum{args.cut_num}_en{args.z0_encoder}"
    if os.path.isfile(f'./logs/{name}.pkl') and not args.overwrite_results:
        print('the results of the following configuration exists \n',
              f'./logs/{name}.pkl')
    else:
        results = []
        for ite in range(args.repeats):
            print(name, ite)
            try:
                acc_matrix, model, start_end_mses = [], None, {}

                scores = []

                for tid, task in enumerate(args.tasks):
                    print(f'------------------------------learning the {tid}-th task: {task}---------------------------')

                    t1 = time.time()

                    if args.baseline in ['joint'] and tid<len(args.tasks)-1:
                        print('')
                        if args.joint_skip:
                            continue #if args.joint_skip:

                    model, _, start_mse, end_mse = main(args, model=model, task_id=tid, motion=args.system == 'motion', mask_id=tid)

                    scores.append(get_model_scores(model))

                    start_end_mses[tid] = [start_mse, end_mse] # record performance at start and end of a task

                    if args.mask:
                        utils.cache_masks(model) # for the modules with masks, register 'stack' into buffer with current mask,Buffers will be stored in state_dict, won’t be returned in model.parameters(), optimizer won’t update them
                        utils.set_num_tasks_learned(model, tid + 1) #assign num_tasks_learned as model's attributes


                    # test the model on previous tasks
                    mses_current_task, mses_current_task_assigned_mask = [], {}
                    for test_id, test_task in enumerate(args.tasks[0:tid + 1]):
                        print(f'---------------testing {test_id}-th task after learning task {tid}: {test_task}--------------')
                        print(f'----Automatic mask selection----')
                        _, best_mse, _, _ = main(args, model=model, task_id=test_id, train_model=False,
                                                 motion=args.system == 'motion', mask_id=None) # set task_id as None if testing without known task id
                        mses_current_task.append(best_mse) # results by auto selected mask
                        mses_current_task_assigned_mask[f'{test_id}'] = [] # results by manually assigned masks
                        if args.mask:
                            for mask_id in range(model.num_tasks_learned):
                                print(f'----testing with mask {mask_id}----')
                                _, best_mse, _, _ = main(args, model=model, task_id=test_id, train_model=False,
                                                         motion=args.system == 'motion',
                                                         mask_id=mask_id)  # set task_id as None if testing without known task id
                                mses_current_task_assigned_mask[f'{test_id}'].append(best_mse)
                    acc_matrix.append(mses_current_task)
                print(f'acc per task {name} is:')
                for m in acc_matrix:
                    print(m)
                print(start_end_mses)
                print('performance with each mask\n',mses_current_task_assigned_mask)
                results.append({'tasks':args.tasks,'acc_mat':acc_matrix,'start_end_mses':start_end_mses})

                torch.cuda.empty_cache()

                if ite == 0:
                    # save results once one run is completed
                    with open(f'./logs/log.txt','a') as f:
                        f.write(name)
                        f.write(f'acc_mat:{acc_matrix}\n,start_end_mses:{start_end_mses}')
                    with open(f'./logs/scores_{name}.pkl', 'wb') as f:
                        pickle.dump(scores, f)
            except Exception as e:
                utils.makedirs('./logs/errors/')
                if ite > 0:
                    name_ = f"{args.baseline}_{ts}_{args.sample_percent_train}_{args.sample_percent_test}_{args.mode}_mask{args.mask}_mode{args.mask_select}_dropMASK{args.dropout_mask}_ts{args.thresholding}_{args.latents}_{args.rec_dims}_{args.ode_dims}_e{args.nepos}_b{args.batch_size}_lr{args.lr}_ite{ite}_cutnum{args.cut_num}_en{args.z0_encoder}"
                    with open(
                            f'./logs/{name_}.pkl', 'wb') as f:
                        pickle.dump(results, f)
                print('error', e)
                name = 'errors/{}'.format(name)
                results = traceback.format_exc()
                print(results)
                print('error happens on \n', name)
                torch.cuda.empty_cache()
                break
        if args.save_results:
            with open(f'./logs/{name}.pkl', 'wb') as f:
                pickle.dump(results, f)
