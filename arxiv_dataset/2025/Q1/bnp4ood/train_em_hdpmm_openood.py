"""Expectation Maximization Hierarchical DPMM for OpenOOD"""
# Python
import argparse
import logging
import sys
# Third Party
import torch
# Modules
import bnp4ood.em_hdpmm as em_hdpmm
from bnp4ood.openood_dataset_utils import setup_dataloaders, get_sufficient_stats, openood_eval


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


logger = logging.getLogger("")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
logger.addHandler(ch)

parser = argparse.ArgumentParser()
# Logging
parser.add_argument("--log_path", type=str, default="em_dpmm.log")
parser.add_argument("--output_filename", type=str, default="em_dpmm.result")
# Model
parser.add_argument("--autowhiten", action="store_true")
parser.add_argument("--autowhiten_factor", type=float, default=1e-7)
parser.add_argument("--autowhiten_dim", type=int, default=0)
parser.add_argument("--usepca", action="store_true")
parser.add_argument("--pca_dim", type=int, default=-1)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--features", type=str, default="vit-b-16")


def main(args):
    # Logging
    fh = logging.FileHandler(args.log_path)
    logger.addHandler(fh)
    logger.info("EM Fitting Hierarchical DPMM on ViT Features for OpenOOD")
    logger.info("--------------------------------------------------------")
    logger.info("Arguments:")
    logger.info(args)

    # Load datasets
    dataset_dict, id_feats, id_labels = setup_dataloaders(
        batch_size=args.batch_size,
        autowhiten=args.autowhiten,
        autowhiten_factor=args.autowhiten_factor,
        autowhiten_dim=args.autowhiten_dim,
        use_pca=args.usepca,
        pca_dim=args.pca_dim,
        features=args.features,
    )
    init_feats, _ = next(iter(dataset_dict["id"]["train"]))
    D = init_feats.shape[-1]

    Nk, sumx, sumxxT, _ = get_sufficient_stats(
        X=id_feats, Y=id_labels,
    )

    # Setup Initial priors
    logging.info("Setting up initial priors...")
    mu0_star = sumx.sum(0) / Nk.sum()
    muk_hat = sumx / Nk[:, None]
    Sigma0_star = (sumxxT.sum(0)
                   - torch.einsum('ki,kj->ij', sumx, muk_hat)
                   - torch.einsum('ki,kj->ij', muk_hat, sumx)
                   + torch.einsum("k,ki,kj->ij", Nk, muk_hat, muk_hat)) / Nk.sum()

    init_nu0 = nu0_star = torch.clamp(Nk.mean(), min=D+1.1)
    init_kappa0 = kappa0_star = torch.tensor(1e-4, device=DEVICE)

    # Run EM
    logging.info("Running EM...")
    lls, kappas, nus = em_hdpmm.em(Nk, sumx, sumxxT, nu0_star, Sigma0_star,
                                   kappa0_star, mu0_star, num_iter=100)
    kappa0_star = kappas[-1]
    nu0_star = nus[-1]
    torch.cuda.empty_cache()
    output = dict(em=dict(lls=lls, kappas=kappas, nus=nus))

    # Calculate Post Params
    logging.info("Calculating posterior params...")
    prior_params = (torch.as_tensor(nu0_star, device=DEVICE),
                    Sigma0_star * (nu0_star - D - 1),
                    torch.as_tensor(kappa0_star, device=DEVICE), mu0_star)
    posterior_params = em_hdpmm.compute_posterior_params(
        Nk, sumx, sumxxT, nu0_star, Sigma0_star, kappa0_star, mu0_star)
    torch.cuda.empty_cache()

    # Evaluate
    logger.info("Evaluating on OpenOOD benchmark")
    params = (prior_params, posterior_params)
    def compute_scores(X, prior_params, posterior_params, return_preds=False):
        return em_hdpmm.compute_scores(prior_params, posterior_params, X,
                                       return_preds=return_preds,
                                       Ns=Nk,
                                      )
    aurocs = openood_eval(dataset_dict, params, compute_scores, model="DPMM")
    output["openood"] = aurocs
    logger.info("EM HDPMM AUROC:")
    for k, v in aurocs.items():
        logger.info(f"{k}: {v}")
    torch.save(output, args.output_filename)


if __name__ == "__main__":
    main(parser.parse_args())