"""Expectation maximization hierarchical DPMM with coupled diagonal covariance for OpenOOD

Script runs the EM algorithm to fit the prior parameters of the diagonal
covariance HDPMM model on the ViT features of the Imagenet 1K dataset, then
evaluates model's OOD detection performance on the OpenOOD benchmark.
"""
# Python
import argparse
import logging
import sys
# Third-party
import torch
# Module
import bnp4ood.em_coupled_diag_hdpmm as cdhdpmm
from bnp4ood.openood_dataset_utils import setup_dataloaders, get_sufficient_stats, openood_eval


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


logger = logging.getLogger("")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
logger.addHandler(ch)

parser = argparse.ArgumentParser()
# Logging
parser.add_argument("--log_path", type=str, default="em_coupled_diag_dpmm.log")
parser.add_argument("--output_filename", type=str, default="em_coupled_diag_dpmm.result")
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
    logger.info("EM Fitting Coupled Diagonal Cov Hierarchical DPMM on ViT Features for OpenOOD")
    logger.info("-----------------------------------------------------------------------------")
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

    Nk, sumx, _, sumxsq = get_sufficient_stats(
        X=id_feats, Y=id_labels,
    )

    # Setup Initial priors
    logging.info("Setting up initial priors...")
    mu0_star = sumx.sum(0) / Nk.sum()
    muk_hat = sumx / Nk[:, None]
    sigmasq0_star = (
        sumxsq.sum(0) + torch.einsum("k,ki->i", Nk, muk_hat ** 2)
        - 2 * (muk_hat * sumx).sum(0)) / Nk.sum()
    init_nu0 = Nk.max() * torch.ones(D, device=DEVICE)
    init_kappa0 = (1. / 5.**2) * torch.ones(D, device=DEVICE)
    init_alpha0 = torch.tensor(3.0, device=DEVICE)
    gammaks = torch.logspace(-1, 0.75, 100, base=10, device=DEVICE)

    # Run EM
    logging.info("Running EM...")
    lls, kappas, nus, alphas = cdhdpmm.em(
        Nk, sumx, sumxsq,
        init_nu0, sigmasq0_star, init_kappa0, mu0_star, init_alpha0,
        num_iter=100, gammaks=gammaks)
    kappa0_star = kappas[-1]
    nu0_star = nus[-1]
    alpha0_star = alphas[-1]
    torch.cuda.empty_cache()
    output = dict(em=dict(lls=lls, kappas=kappas, nus=nus, alphas=alphas))

    # Calculate Post Params
    logging.info("Calculating posterior params...")
    centers, _, logweights = cdhdpmm.gammak_centers_weights(alpha0_star.to(DEVICE), gammaks)
    prior_params = (torch.as_tensor(nu0_star, device=DEVICE),
                    centers[:, None] * sigmasq0_star,
                    torch.as_tensor(kappa0_star, device=DEVICE),
                    mu0_star,
                    torch.exp(logweights), logweights)
    posterior_params = cdhdpmm.compute_posterior_params(
        Nk, sumx, sumxsq,
        torch.as_tensor(nu0_star, device=DEVICE),
        sigmasq0_star, torch.as_tensor(kappa0_star, device=DEVICE), mu0_star,
        torch.as_tensor(alpha0_star, device=DEVICE), gammaks)
    torch.cuda.empty_cache()

    # Evaluate
    logger.info("Evaluating on OpenOOD benchmark ")
    params = (prior_params, posterior_params)
    def compute_scores(X, prior_params, posterior_params, return_preds=False):
        return cdhdpmm.compute_scores(prior_params, posterior_params, X,
                                      return_preds=return_preds,
                                      Ns=Nk,
                                     )
    aurocs = openood_eval(dataset_dict, params, compute_scores, model="Coupled Diag DPMM")
    output["openood"] = aurocs
    logger.info("EM Coupled Diagonal HDPMM AUROC:")
    for k, v in aurocs.items():
        logger.info(f"{k}: {v}")
    torch.save(output, args.output_filename)


if __name__ == "__main__":
    main(parser.parse_args())
