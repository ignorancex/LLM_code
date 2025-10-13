from glob import glob
import torch
import pandas as pd
import numpy as np

from argparse import ArgumentParser


def all_correlations(x, y):
    N = x.shape[0]

    sigma_x = torch.std(x, dim=0) + 1e-10
    sigma_y = torch.std(y, dim=0) + 1e-10

    x_b = (x - torch.mean(x, dim=0)) / sigma_x
    del x
    del sigma_x
    y_b = (y - torch.mean(y, dim=0)) / sigma_y

    corrs = (x_b.T @ y_b) / N
    
    return corrs



def wmppc(runA, runB):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    target_dtype = torch.float32
    layersA = sorted(glob(f'sae_features/{runA}/layer*.pt'))
    layersB = sorted(glob(f'sae_features/{runB}/layer*.pt'))
    print(layersA, layersB)
    wMPPC_list = []
    layer_length = []
    layerwise_wmppc_global = []

    with torch.no_grad():
        for layerA in layersA:
            filesA = [layerA]
            t1 = torch.vstack([torch.load(f, map_location=device) for f in filesA]).to(target_dtype).detach()
            sum_A = t1.sum(axis=0, dtype=torch.float32)
            rho_list = []
            layerA_wise_mppc = []
            for layerB in layersB:  
                filesB = [layerB]
                t2 = torch.vstack([torch.load(f, map_location=device) for f in filesB]).to(target_dtype)
                res = all_correlations(t1, t2)
                del t2
                rhoAB = res.max(axis=1)[0]
                rho_list.append(rhoAB)
                
                wMPPC_AB = (rhoAB * sum_A).sum()/(sum_A).sum()
                layerA_wise_mppc.append(wMPPC_AB)
                del res
                del rhoAB
            del t1

            max_corrs_A = torch.vstack(rho_list).max(axis=0)[0]
            wMPPC_A = (max_corrs_A * sum_A).sum()/(sum_A).sum()
            wMPPC_list.append(wMPPC_A.cpu().numpy())
            layer_length.append(len(max_corrs_A))
            layerwise_wmppc_global.append(layerA_wise_mppc)
            del sum_A
            del max_corrs_A

        last = pd.DataFrame(layerwise_wmppc_global).iloc[-1,-1].item()
        wMPPC = np.sum([(w*c)/np.sum(layer_length) for w,c in zip(layer_length, wMPPC_list)])
        
        return wMPPC, last
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--source', default=None, type=str)
    parser.add_argument('--target', default=None, type=str)
    args = parser.parse_args()

    all, last = wmppc(args.source, args.target)

    print(f"wMPPC : {args.source} -> {args.target}")
    print(f"All Layers : {all} \t|\t Last layer : {last}")