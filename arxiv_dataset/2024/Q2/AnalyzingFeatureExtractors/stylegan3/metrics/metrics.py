import numpy as np
import torch
from metric_utils import FeatureStats
import scipy
from precision_recall import compute_distances
from glob import glob
import json

def compute_feature_stats_for_embeddings(feature_path):
    features = np.load(feature_path)
    num_items = len(features)
    stats = FeatureStats(capture_all=True, capture_mean_cov=True,max_items=num_items)    
    stats.append(features)
    return stats


def compute_fid(stats_1,stats_2):
    mu_1, sigma_1 = stats_1.get_mean_cov()
    mu_2, sigma_2 = stats_2.get_mean_cov()

    m = np.square(mu_2 - mu_1).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_2, sigma_1), disp=False) # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_2 + sigma_1 - s * 2))
    return float(fid)

def compute_kid(stats_1,stats_2,num_subsets=100, max_subset_size=1000):
    features_1= stats_1.get_all()
    features_2 = stats_2.get_all()

    n = features_1.shape[1]
    m = min(min(features_1.shape[0], features_2.shape[0]), max_subset_size)
    t = 0
    for _subset_idx in range(num_subsets):
        x = features_2[np.random.choice(features_2.shape[0], m, replace=False)]
        y = features_1[np.random.choice(features_1.shape[0], m, replace=False)]
        a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
        b = (x @ y.T / n + 1) ** 3
        t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
    kid = t / num_subsets / m
    return float(kid)

def compute_pr(stats_1,stats_2, nhood_size=3, row_batch_size=10000, col_batch_size=10000,device="cuda:0",num_gpus=1,rank=0):
    features_1= stats_1.get_all_torch().to(torch.float16).to(device)
    features_2 = stats_2.get_all_torch().to(torch.float16).to(device)

    results = dict()
    for name, manifold, probes in [('precision', features_1, features_2), ('recall', features_2, features_1)]:
        kth = []
        for manifold_batch in manifold.split(row_batch_size):
            dist = compute_distances(row_features=manifold_batch, col_features=manifold, num_gpus=num_gpus, rank=rank, col_batch_size=col_batch_size)
            kth.append(dist.to(torch.float32).kthvalue(nhood_size + 1).values.to(torch.float16) if rank == 0 else None)
        kth = torch.cat(kth) if rank == 0 else None
        pred = []
        for probes_batch in probes.split(row_batch_size):
            dist = compute_distances(row_features=probes_batch, col_features=manifold, num_gpus=num_gpus, rank=rank, col_batch_size=col_batch_size)
            pred.append((dist <= kth).any(dim=1) if rank == 0 else None)
        results[name] = float(torch.cat(pred).to(torch.float32).mean() if rank == 0 else 'nan')
    return results['precision'], results['recall']

def compute_scores(feats_root = "<your root path>"):
    
    target_datas = ["ffhq"]
    source_datas = ["celebaHQ","projectedgan","stylegan2-ffhq"]
    results = []
    for feat_type in ["arcface","arcface_multilayer","arcface_spatial","clip","inception","dino","dinoV2","vit"]:
        target_feature_paths = [feats_root+x+"\\"+feat_type+"_feats.npy" for x in target_datas]
        source_feature_paths = [feats_root+x+"\\"+feat_type+"_feats.npy" for x in source_datas]

        for target_feat_path in target_feature_paths:
            target_data_name = target_feat_path.split("\\")[-2]
            stats_target = compute_feature_stats_for_embeddings(target_feat_path)

            for source_feat_path in source_feature_paths:
                source_data_name = source_feat_path.split("\\")[-2]
                stats_source = compute_feature_stats_for_embeddings(source_feat_path)
                print(target_data_name,source_data_name)
                fid_res = compute_fid(stats_target,stats_source)
                print("FID - {} - {} - {} : {}".format(feat_type,target_data_name,source_data_name,fid_res))
                results.append({"metric":"FID","feat_type":feat_type,"target_data_name":target_data_name,"source_data_name":source_data_name,"score":fid_res})

                kid_res = compute_kid(stats_target,stats_source)
                print("KID - {} - {} - {} : {}".format(feat_type,target_data_name,source_data_name,kid_res))
                results.append({"metric":"KID","feat_type":feat_type,"target_data_name":target_data_name,"source_data_name":source_data_name,"score":kid_res})

                pr_res = compute_pr(stats_target,stats_source)
                print("P&R - {} - {} - {} : {}".format(feat_type,target_data_name,source_data_name,pr_res))
                results.append({"metric":"P&R","feat_type":feat_type,"target_data_name":target_data_name,"source_data_name":source_data_name,"score":pr_res})
                
                print(flush=True)

    with open('<your path>.json', 'w') as fout:
        json.dump(results, fout)




if __name__ == "__main__":
    compute_scores()
    
    
    
