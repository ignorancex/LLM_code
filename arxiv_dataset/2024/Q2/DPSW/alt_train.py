import sys,traceback
import random
import numpy as np
import torch
from config import args
from load_data import DataLoader
from alt_obj_func import *
from model.dr_cfrnet import CFR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score

import time

print(args)

def inverse_scale(scaler, y):
    y_numpy = y.numpy()
    y_numpy = scaler.inverse_transform(y_numpy)
    y = torch.tensor(y_numpy)
    return y

def main(idx,heavi_k):
    torch.manual_seed(args.train_seed)
    torch.cuda.manual_seed(args.train_seed)
    np.random.seed(args.train_seed)
    random.seed(args.train_seed)

    #use_gpu = not args.no_gpu and (torch.backends.mps.is_available() or torch.cuda.is_available())
    use_gpu = not args.no_gpu and (torch.cuda.is_available())
    if torch.cuda.is_available():
        device_type = "cuda"
    # elif torch.backends.mps.is_available():
    #     device_type = "mps"
    else:
        device_type = "cpu"

    device = torch.device(type= device_type)#, index=args.gpu)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    print('Data type: {}'.format(args.data))
    print('Dataset index: {}'.format(idx))

    Data = DataLoader(args=args, idx=idx)
    tr_loader = Data.tr_loader
    val_loader = Data.val_loader ### validation data are used for MSE loss evaluation
    val2_loader = Data.val2_loader ### validation data, including true Y0 and Y1 
    te_loader = Data.te_loader
    in_loader = Data.in_loader ### training + validation data, including true Y0 and Y1. validation data can be used for hyperparameter tuning based on true PEHE

    model = CFR(Data.in_dim, args).to(device)
    p = Data.p_tr # p = n1 / (n0 + n1): proportion of treated individuals in training data

    params_pi0 = model.propensity_net2.parameters()
    params_rest = model.params # other parameters
    if args.optim == "SGD":
        optimizer1 = torch.optim.SGD(params_pi0, lr=args.lrp)
        optimizer2 = torch.optim.SGD(params_rest, lr=args.lr)        
    elif args.optim == "Adam":
        optimizer1 = torch.optim.Adam(params_pi0, lr=args.lrp, weight_decay=args.weight_decay)
        optimizer2 = torch.optim.Adam(params_rest, lr=args.lr, weight_decay=args.weight_decay)        
    else:
        print("Error: optimizer = " + str(args.optim) + " is not defined", file=sys.stderr)
        sys.exit(1)           

    loss_func = 'l2'
    imbalance_func = args.imbalance_func
    estim_pareto_params = args.Pareto_est
    heavi_model = args.heavi_method
    weight_method = args.weight_method
    reg_alpha = args.reg_alpha
    reg_soft = args.reg_soft

    print('Training started ...')
    torch.autograd.set_detect_anomaly(True)
    num_of_subepochs = 100; num_of_epochs = int(args.epochs / num_of_subepochs)
    pehe_hist = []
    for epoch in range(num_of_epochs):
        # freeze other parameters than pi_0
        for param in params_rest:
            param.requires_grad = False
        for param in params_pi0:
            param.requires_grad = True
        # learn pi_0
        for subepoch1 in range(num_of_subepochs):
            for ind, a, x, y in tr_loader:
                ind, a, x, y = ind.to(device), a.to(device), x.to(device), y.to(device)
                a = a.reshape((a.shape[0], 1)); y = y.reshape((y.shape[0], 1))
                y_hat, pi_0 = model(x, a)   #pi_0 for cross entropy loss
                loss1 = obj_pi0(pi_0)
                loss1.backward()
                optimizer1.step()
                optimizer1.zero_grad()
            print("loss1: " + str(loss1.cpu().detach().numpy()))

        # freeze only pi_0
        for param in params_rest:
            param.requires_grad = True
        for param in params_pi0:
            param.requires_grad = False
        # learn other parameters
        for subepoch2 in range(num_of_subepochs):
            for ind, a, x, y in tr_loader:
                ind, a, x, y = ind.to(device), a.to(device), x.to(device), y.to(device)
                a = a.reshape((a.shape[0], 1)); y = y.reshape((y.shape[0], 1))
                y_hat, pi_0 = model(x, a)   #pi_0 for cross entropy loss
                x_encC = model.encodeC(x)
                pi_0 = model.propensity_net2.propensity(pi_0)
                w = calc_weights(a, p, pi_0, weight_method, reg_soft, estim_pareto_params, heavi_model, heavi_k)
                loss2 = objective(y, y_hat, x_encC, a, p, w, reg_alpha, loss_func=loss_func, imbalance_func=imbalance_func)
                loss2.backward()
                optimizer2.step()
                optimizer2.zero_grad()
            print("loss2: " + str(loss2.cpu().detach().numpy()))
            reg_alpha = reg_alpha * args.reg_decay ** subepoch2

        # evaluate PEHE on test data
        pehe = []
        taus_orig = []
        pehe_orig = []
        if loss_func == 'ce': 
            tau_np = []
            tau_hat_np = []

        model.eval()
        with torch.no_grad():
            for ind, x, y0, y1 in te_loader:
                ind, x, y0, y1 = ind.to(device), x.to(device), y0.to(device), y1.to(device)
                y0 = y0.reshape((y0.shape[0], 1)); y1 = y1.reshape((y1.shape[0], 1))
                y0_hat = model.predict(x, 0) ## predict Y0 by model(x, 0)
                y1_hat = model.predict(x, 1) ## predict Y1 by model(x, 1)
                if loss_func == 'ce':
                    y0_hat = soft_max(y0_hat)
                    y1_hat = soft_max(y1_hat)
                    tau_hat = y1_hat.argmax(dim=1) - y0_hat.argmax(dim=1)
                else:
                    tau_hat = y1_hat - y0_hat
                tau = y1 - y0
                pehe.append(torch.square(tau_hat - tau)) 
                data_scale = Data.scalerY.scale_.item() if loss_func != 'ce' else 1.0
                taus_orig.append(torch.square(data_scale*tau)) 
                pehe_orig.append(torch.square(data_scale*(tau_hat - tau))) 
                if loss_func == 'ce':
                    tau_hat_np.append(tau_hat.cpu().detach().numpy())
                    tau_np.append(tau.cpu().detach().numpy())

        pehe = torch.stack(pehe)
        mean_sqrt_pehe = torch.sqrt(torch.mean(pehe)).cpu().detach().numpy()
        print("test_pehe: " + str(mean_sqrt_pehe))
        if loss_func == 'ce':
            tau_np = np.array(tau_np).reshape(len(tau_hat_np[0]))
            tau_hat_np = np.array(tau_hat_np).reshape(len(tau_hat_np[0]))
            print("test accuracy: " + str(accuracy_score(tau_np, tau_hat_np)))
            print("macro f1: " + str(f1_score(tau_np, tau_hat_np, average='macro')))
            print("micro f1: " + str(f1_score(tau_np, tau_hat_np, average='micro')))

        pehe_orig = torch.cat(pehe_orig, dim=0)
        taus_orig = torch.cat(taus_orig, dim=0)
        mean_sqrt_pehe_orig = torch.sqrt(torch.mean(pehe_orig)).cpu().detach().numpy()
        mean_sqrt_taus_orig = torch.sqrt(torch.mean(taus_orig)).cpu().detach().numpy()
        print("test_pehe_orig: " + str(mean_sqrt_pehe_orig))
        print("test_pehe_orig (scaled): " + str(mean_sqrt_pehe_orig/mean_sqrt_taus_orig))

        patience = 5; frequency = 2
        pehe_hist.append(mean_sqrt_pehe_orig)

        model.train()
    print('Training ended')
    if args.data == 'synthH':
        W_dim = model.encoderA.state_dict()['sequential.0.weight'].shape[1]
        V_dim = int(args.data_index)
        WA_ind = np.arange(0, V_dim); WA_ind_ = np.arange(V_dim, W_dim)
        WB_ind = np.arange(V_dim, 2*V_dim); WB_ind_ = np.hstack((np.arange(0, V_dim), np.arange(2*V_dim, W_dim)))
        WC_ind = np.arange(2*V_dim, 3*V_dim); WC_ind_ = np.hstack((np.arange(0, 2*V_dim), np.arange(3*V_dim, W_dim)))
        WA_mean = torch.mean(torch.abs(model.encoderA.state_dict()['sequential.0.weight'][:,WA_ind]))
        WA_mean_ = torch.mean(torch.abs(model.encoderA.state_dict()['sequential.0.weight'][:,WA_ind_]))        
        WB_mean = torch.mean(torch.abs(model.encoderB.state_dict()['sequential.0.weight'][:,WB_ind]))
        WB_mean_ = torch.mean(torch.abs(model.encoderB.state_dict()['sequential.0.weight'][:,WB_ind_]))           
        WC_mean = torch.mean(torch.abs(model.encoderC.state_dict()['sequential.0.weight'][:,WC_ind]))
        WC_mean_ = torch.mean(torch.abs(model.encoderC.state_dict()['sequential.0.weight'][:,WC_ind_]))
        return mean_sqrt_pehe_orig, (WA_mean - WA_mean_)/WA_mean_, (WB_mean - WB_mean_)/WB_mean_, (WC_mean - WC_mean_)/WC_mean_
    return mean_sqrt_pehe_orig

if __name__ == '__main__':
    dataname = args.data
    indices = args.data_index.strip().split(',')
    indices = [int(idx) for idx in indices]
    flag_notall = indices[0]
    if args.data == 'news':
        num_of_datasets = 50
    elif args.data == 'LBIDD':
        num_of_datasets = 20

    if flag_notall:
        num_of_datasets = len(indices)
    else:
        indices = range(1, num_of_datasets + 1)
    
    if args.data == 'synthH':
        num_of_datasets = 20
        indices = [args.data_index] * num_of_datasets
        wa = []; wb = []; wc = []

    success_idx= []
    res_sqrt_pehe_orig = []
    val_res_sqrt_pehe_orig = []
    val_res_loss_orig = []


    for i in range(len(indices)):
        heavi_k = args.heavi_k
        if args.data == 'synthH': 
            tp, wa_, wb_, wc_ = main(indices[i], heavi_k)
            wa.append(wa_); wb.append(wb_); wc.append(wc_)
        else:
            tp = main(indices[i], heavi_k)
        res_sqrt_pehe_orig.append(tp)
    res_sqrt_pehe_orig = np.array(res_sqrt_pehe_orig)
    if args.data == 'synthH': 
        wa_np = np.array(wa)
        wb_np = np.array(wb)
        wc_np = np.array(wc)
    if num_of_datasets > 1:
        print("PEHE (Mean +- Standard Deviation)")
        print(str(np.mean(res_sqrt_pehe_orig)) + " +- " + str(np.std(res_sqrt_pehe_orig)))
        if args.data == 'synthH': 
            print("Parameter value differences of trained encoders Gamma, Delta, and Upsilon:")
            print(str(np.mean(wa_np)) + " +- " + str(np.std(wa_np)))
            print(str(np.mean(wb_np)) + " +- " + str(np.std(wb_np)))
            print(str(np.mean(wc_np)) + " +- " + str(np.std(wc_np)))
    else:
        print(res_sqrt_pehe_orig)

    print("---setting---")
    print(vars(args))
    print("------------")

    if args.save_result:
        np.savetxt(f"result/log/{dataname}_{indices[0]}_{indices[-1]}_{args.reg_beta}-{args.Pareto_est}_{args.heavi_method}_{args.enc_h_dim}_{args.propensity_net_type}.txt", np.concatenate([res_sqrt_pehe_orig.reshape(-1,1),val_res_sqrt_pehe_orig.reshape(-1,1),val_res_loss_orig.reshape(-1,1)], axis=1))
        with open(f"{dataname}.dat", 'a') as f:
            print(f"{dataname}.dat")
            print('args', (args.reg_alpha, args.reg_beta, args.weight_decay, args.Pareto_est, args.heavi_method, args.reg_soft, args.enc_h_dim, args.oh_h_dim, args.epochs, args.batch_size, args.propensity_net_type), file=f)
            print('test_mean_pehe_orig', str(np.mean(res_sqrt_pehe_orig)), file=f)
            print('test_std_pehe_orig', str(np.std(res_sqrt_pehe_orig)), file=f)
        np.savetxt(f"result/res_sqrt_pehe_orig_{dataname}_{indices[0]}_{indices[-1]}_{args.reg_beta}-{args.Pareto_est}_{args.heavi_method}_{args.enc_h_dim}_{args.propensity_net_type}.txt", res_sqrt_pehe_orig)
