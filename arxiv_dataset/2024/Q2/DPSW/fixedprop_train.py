import sys,traceback
import random
import numpy as np
import torch
import torch.nn as nn
from config import args
from load_data import DataLoader
from alt_obj_func import *
from model.dr_cfrnet import CFR
# from model.cfrnet import CFR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score

import time

print(args)

def load_weights(ind):
    if args.data == 'news':
        file_name = 'data/NEWS_csv/pipw_tr.csv' 
        scale_ = 1.
    elif args.data == 'LBIDD':
        file_name = 'data/LBIDD/pipw_tr.csv'
        scale_ = 1.
    W_tr = np.loadtxt(file_name, delimiter=',')
    ind = ind.astype(int)
    w = W_tr[ind, int(args.data_index)]
    return torch.from_numpy(scale_ * w.reshape((w.shape[0], 1)))

class CFR(nn.Module):
    def __init__(self, in_dim, args):
        super().__init__()
        if args.imbalance_func in ['lin_disc', 'mmd2_lin']:
            enc_activation = None
        else:
            enc_activation = nn.ELU(inplace=True)

        self.encoderB = MLP(
            num_layers=args.enc_num_layers,
            in_dim=in_dim,
            h_dim=args.enc_h_dim,
            out_dim=args.enc_out_dim,
            activation=enc_activation,
            dropout=args.enc_dropout,
        )
        self.encoderC = MLP(
            num_layers=args.enc_num_layers,
            in_dim=in_dim,
            h_dim=args.enc_h_dim,
            out_dim=args.enc_out_dim,
            activation=enc_activation,
            dropout=args.enc_dropout,
        )

        self.bool_outhead = args.bool_outhead
        ## CFR (Shalit+; ICML2017)
        if self.bool_outhead: 
            self.outhead_Y1 = MLP(
                num_layers=args.oh_num_layers,
                in_dim=2*args.enc_out_dim,
                h_dim=args.oh_h_dim,
                out_dim=args.oh_out_dim,
                dropout=args.oh_dropout
            )
            self.outhead_Y0 = MLP(
                num_layers=args.oh_num_layers,
                in_dim=2*args.enc_out_dim,
                h_dim=args.oh_h_dim,
                out_dim=args.oh_out_dim,
                dropout=args.oh_dropout
            )

        ## BNN (Johansson+; ICML2016)
        else:
            self.decoder = MLP(
                num_layers=args.oh_num_layers,
                in_dim=args.enc_out_dim + 1,
                h_dim=args.oh_h_dim,
                out_dim=args.oh_out_dim, 
                dropout=args.oh_dropout
            )

            self.params = (
                list(self.encoder.parameters())
                + list(self.decoder.parameters())
            )

    def encodeB(self, x):
        x_encB = self.encoderB(x)
        return x_encB
    def encodeC(self, x):
        x_encC = self.encoderC(x)
        return x_encC

    def forward(self, x, a):
        x_encB = self.encoderB(x)
        x_encC = self.encoderC(x)
        x_enc = torch.cat([x_encB, x_encC], 1)
        if self.bool_outhead:
            _t_ind = torch.where(a == 1)[0] ## a: treatment vector in torch.Tensor
            _c_ind = torch.where(a == 0)[0] 
            _ind = torch.argsort(torch.concatenate([_t_ind, _c_ind], 0))  
            y1_hat = self.outhead_Y1(x_enc[_t_ind])
            y0_hat = self.outhead_Y0(x_enc[_c_ind])
            y_hat = torch.cat([y1_hat, y0_hat])[_ind]
        else:
            y_hat = self.decoder(torch.cat((x_enc, a.reshape((x_enc.shape[0], 1))), 1))
        return y_hat

    def predict(self, x, a_val):
        x_encB = self.encoderB(x)
        x_encC = self.encoderC(x)
        x_enc = torch.cat([x_encB, x_encC], 1)
        if self.bool_outhead:
            if a_val == 0:
                return self.outhead_Y0(x_enc)
            else:
                return self.outhead_Y1(x_enc)
        else:
            if a_val == 0:
                return self.decoder(torch.cat((x_enc, torch.zeros(x.shape[0]).reshape((x_enc.shape[0], 1))), 1))
            else:
                return self.decoder(torch.cat((x_enc, torch.ones(x.shape[0]).reshape((x_enc.shape[0], 1))), 1)) 

class MLP(nn.Module):
    def __init__(
        self,
        in_dim,
        num_layers,
        h_dim,
        out_dim,
        activation=torch.nn.ELU(inplace=True),
        dropout=0.2,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.num_layers = num_layers
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.activation = activation
        self.dropout = dropout

        bool_nonlin = False if self.activation is None else True
        layers = []
        for i in range(num_layers - 1):
            layers.extend(
                self._layer(
                    h_dim if i > 0 else in_dim,
                    h_dim,
                    bool_nonlin,
                )
            )
        layers.extend(self._layer(h_dim, out_dim, False))

        self.sequential = torch.nn.Sequential(*layers)

    def _layer(self, in_dim, out_dim, activation=True):
        if activation:
            return [
                torch.nn.Linear(in_dim, out_dim),
                self.activation,
                torch.nn.Dropout(self.dropout),
            ]
        else:
            return [
                torch.nn.Linear(in_dim, out_dim),
            ]

    def forward(self, x):
        out = self.sequential(x)
        return out

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

    use_gpu = not args.no_gpu and (torch.cuda.is_available())
    if torch.cuda.is_available():
        device_type = "cuda"
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

    if args.optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.optim == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)   
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
    num_of_epochs = int(args.epochs)
    pehe_hist = []
    for epoch in range(num_of_epochs):
        for ind, a, x, y in tr_loader:
            ind, a, x, y = ind.to(device), a.to(device), x.to(device), y.to(device)

            a = a.reshape((a.shape[0], 1)); y = y.reshape((y.shape[0], 1))
            y_hat = model(x, a)  
            x_encC = model.encodeC(x)
            w = load_weights(ind.detach().numpy())
            loss = objective(y, y_hat, x_encC, a, p, w, reg_alpha, loss_func=loss_func, imbalance_func=imbalance_func)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print("loss: " + str(loss.cpu().detach().numpy()))
        reg_alpha = reg_alpha * args.reg_decay ** epoch

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
                    np.set_printoptions(threshold=np.inf)
                    #sys.exit()
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
        #print("test_pehe_orig (scaled): " + str(mean_sqrt_pehe_orig/mean_sqrt_taus_orig))

        pehe_hist.append(mean_sqrt_pehe_orig)

        model.train()
    print('Training ended')

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
        print(str(i) + ': Current result ')
        print(str(np.mean(res_sqrt_pehe_orig)) + " +- " + str(np.std(res_sqrt_pehe_orig)))

    res_sqrt_pehe_orig = np.array(res_sqrt_pehe_orig)
    if args.data == 'synthH': 
        wa_np = np.array(wa)
        wb_np = np.array(wb)
        wc_np = np.array(wc)

    if num_of_datasets > 1:
        print(str(np.mean(res_sqrt_pehe_orig)) + " +- " + str(np.std(res_sqrt_pehe_orig)))
        if args.data == 'synthH': 
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
