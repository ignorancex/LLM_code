"""A minimal / pedagogical implementation of the central flow for gradient descent."""

import os

import torch
import tyro
from tqdm import trange
from pathlib import Path
import random
import string

from src.architectures import MLP
from src.sdcp import SDCPSolver
from src.datasets import CIFAR10
from src.eig_solvers import WarmStartEigSolver
from src.functional import FunctionalModel
from src.loss_function import LossFunction, SupervisedLossFunction
from src.utils import apply_to_pairs, mat_to_upper, upper_to_mat, get_upper_to_mat_metric, save_pytree

from torch.utils._pytree import tree_map
import torch.nn.functional as F


def main(
    lr: float,
    steps: int,
    device: str = "cuda",
    nsubsteps_central: int = 4,
    nsubsteps_stable: int = 10,
    n_eig: int = 2,
    seed: int = 0,
    save_freq: int = 10,
):
    # specify the dataset and architecture
    data = CIFAR10(criterion="mse", n=1000, classes=4)
    # TODO update doc to reflect this
    # arch = CNN()
    arch = MLP()
    
    torch.manual_seed(seed)

    # generate random experiment id
    expid = ''.join(random.choices(string.ascii_letters, k=8)) # random 8 letter identifier
    
    # create output directory
    outdir = Path(f"experiments/{expid}")
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"output directory: {outdir}")

    # load the dataset
    dataset = data.load(device=device)

    # instantiate model as pytorch module
    model = arch.create(dataset.input_shape, dataset.output_dim).to(device)

    # make everything functional. 'model_fn' is a functional version of the network; 'w' are the initial weights
    w, model_fn = FunctionalModel.make_functional(model)

    # put together the training objective
    loss_fn = SupervisedLossFunction(
        model_fn=model_fn, criterion=dataset.criterion_fn, batches=dataset.trainset
    )

    # solvers for computing Hessian eigenvectors using warm-started LOBPCG
    eig_solver_discrete = WarmStartEigSolver(loss_fn, w, initial_neigs=n_eig)
    eig_solver_central = WarmStartEigSolver(loss_fn, w, initial_neigs=n_eig)
    eig_solver_stable = WarmStartEigSolver(loss_fn, w, initial_neigs=n_eig)

    # initialize discrete optimizer, central flow, and stable flow all at the same point
    w_discrete, w_central, w_stable = w.clone(), w.clone(), w.clone()

    # store experimental data from all steps as a hierarchical dict where leaves are
    # tensors that contain time series
    data = {}

    # main training loop - run central flow and discrete optimizer in parallel
    for step in trange(steps):
        # store experimental data from this step as a hierarchical dict where
        # leaves are floats or tensors
        datum = {}

        # compute updates for discrete optimizer, central flow, and stable flow
        next_w_discrete, datum["discrete"] = step_discrete(
            w_discrete, loss_fn, lr, eig_solver_discrete
        )
        next_w_central, datum["central"], sigma_evecs = step_central(
            w_central, loss_fn, lr, eig_solver_central, nsubsteps_central
        )
        next_w_stable, datum["stable"] = step_stable(
            w_stable, loss_fn, lr, eig_solver_stable, nsubsteps_stable
        )
        
        # record distances from both flows to discrete optimizer
        datum["distances"] = dict(
            central=(w_discrete - w_central).norm().item(),
            stable=(w_discrete - w_stable).norm().item()
        )

        # record displacement from central flow to discrete optimizer along eigenvectors of Sigma
        datum["delta"] = sigma_evecs.t().mv(w_discrete - w_central) ** 2

        # apply updates
        w_discrete, w_central, w_stable = next_w_discrete, next_w_central, next_w_stable

        # concatenate datum from this step to the overall data
        data = concatenate_data(data, datum)

        # save all experimental data every so often
        if step % save_freq == 0 or step == steps - 1:
            save_pytree(data, outdir / "data.pt")


def step_discrete(
    w: torch.tensor,
    loss_fn: LossFunction,
    lr: float,
    eig_solver: WarmStartEigSolver,
):
    """Run gradient descent for one step.

    Returns:
      tensor: weights at the next step
      dict: dictionary storing logged data
    """
    gradient, loss = loss_fn.grad_and_value(w)
    eigs = eig_solver.update(w)[0]
    w_next = w - lr * gradient
    out = dict(train_loss=loss, hessian_eigs=eigs)
    return w_next, out


def step_central(
    w: torch.tensor,
    loss_fn: LossFunction,
    lr: float,
    eig_solver: WarmStartEigSolver,
    nsubsteps: int,
    sdcp_threshold: float = 1.95
):
    """Discretize the GD central flow for one unit of time.

    Returns:
      tensor: weights after one unit of time
      dict: dictionary storing logged data
    """
    dt = 1.0 / nsubsteps  # the time increment of each substep
    sdcp_solver = SDCPSolver()

    for k in range(nsubsteps):
        out = dict()  # data to log goes in here

        # compute loss and gradient
        nabla_L, loss = loss_fn.grad_and_value(w)

        # re-compute top eigenvalues and eigenvectors using warm-started LOBPCG
        eigs, U, _ = eig_solver.update(w)

        out.update(train_loss=loss, hessian_eigs=eigs)

        # the number of eigenvalues that are near the critical threshold
        crit_k = (eigs > sdcp_threshold / lr).sum()

        # if all eigenvalues are well below 2/LR, run gradient flow
        if crit_k == 0:
            w = w - dt * lr * nabla_L
            out.update(predicted_loss=loss, sigma_eigs=torch.zeros(len(eigs)))
            Sigma_U = U
        # but if at least one eigenvalue is near-critical ...
        elif crit_k > 0:
            # extract the critical eigenvalues and eigenvectors
            crit_eigs, crit_U = eigs[:crit_k], U[:, :crit_k]

            # compute d/dw u'H(w)v for all pairs (u, v) of critical eigenvectors
            nabla_H = apply_to_pairs(
                lambda u, v: loss_fn.D(w, 3, u, v), crit_U
            )  # [crit_k + 1 choose 2, nparam]

            # compute X, the covariance of the oscillations within the basis crit_U, so that Sigma = crit_U X crit_U'
            dw_dt_stable = -lr * nabla_L     # [nparam]
            dw_dt_X = -0.5 * lr * nabla_H.T  # [nparam, crit_k choose 2]
            dH_dt_stable = upper_to_mat(nabla_H @ dw_dt_stable)  # [crit_k, crit_k]
            dH_dt_X = -upper_to_mat(nabla_H @ dw_dt_X)           # [crit_k, crit_k, crit_k, crit_k]
            
            I = torch.eye(crit_k).to(w.device)
            next_residual_stable = (2/lr)*I - (torch.diag(crit_eigs) + dt*dH_dt_stable)  # [crit_k, crit_k]
            next_residual_X = dt*dH_dt_X                                                 # [crit_k, crit_k, crit_k, crit_k]
            
            X = sdcp_solver(A=next_residual_stable.cpu().numpy(), B=next_residual_X.cpu().numpy())
            X = torch.from_numpy(X).float().cuda()  # [crit_k, crit_k]
            X_vech = mat_to_upper(X)                # [crit_k+1 choose 2]
            
            # compute next w
            metric = get_upper_to_mat_metric(crit_k)                # [crit_k+1 choose 2]
            w = w + dt * (dw_dt_stable + dw_dt_X@(metric*X_vech))   # [nparams]

            # write Sigma in its diagonal basis, so that Sigma = SigmaU Sigma_eig Sigma_U'
            X_eig, X_U = eigh(X)
            Sigma_eig, Sigma_U = X_eig, crit_U @ X_U

            # compute predicted loss
            predicted_loss = loss + Sigma_eig.sum().item() / lr

            # pad with zeros on the right as needed
            if len(Sigma_eig) < len(eigs):
                Sigma_eig = F.pad(Sigma_eig, (0, len(eigs) - len(Sigma_eig)))
                Sigma_U = torch.hstack((Sigma_U, U[:,crit_k:]))
            
            out.update(predicted_loss=predicted_loss, sigma_eigs=Sigma_eig)


        # we want to keep the out from the first substep
        if k == 0:
            out_first_substep = out

    return w, out_first_substep, Sigma_U


def step_stable(
    w: torch.tensor,
    loss_fn: LossFunction,
    lr: float,
    eig_solver: WarmStartEigSolver,
    nsubsteps: int
):
    """Discretize the GD stable flow (i.e. the gradient flow) for one unit of time.
    
    Returns:
      tensor: weights after one unit of time
      dict: dictionary storing logged data
    """
    dt = 1.0 / nsubsteps  # the time increment of each substep

    for k in range(nsubsteps):
        out = dict()
        
        # compute loss and gradient
        gradient, loss = loss_fn.grad_and_value(w)
        out.update(train_loss=loss)
        
        # re-compute top Hessian eigs if this is the first substep
        if k == 0:
            out.update(hessian_eigs=eig_solver.update(w)[0])
        
        # update weights
        w = w - dt * lr * gradient
        
        # we want to keep the out from the first substep
        if k == 0:
            out_first_substep = out
        
    return w, out_first_substep


"""Helper functions"""


def concatenate_data(data, new_datum):
    """Append data from the latest step to the overall time series."""

    def ensure_tensor(value):
        if isinstance(value, float):
            return torch.tensor(value, device='cpu')
        return value.to('cpu')

    if data == {}:  # if the existing data is empty, i.e. if this is the first step
        return tree_map(lambda new: ensure_tensor(new).unsqueeze(0), new_datum)
    else:
        return tree_map(
            lambda old, new: torch.concatenate((old, ensure_tensor(new).unsqueeze(0))),
            data,
            new_datum,
        )

def eigh(A):
    """Compute eigenvectors/eigenvalues and return them in descending order."""
    eig, U = torch.linalg.eigh(A)  # this returns them in ascending order
    eig, U = eig.flip(-1), U.flip(-1)  # so we flip them
    return eig, U


if __name__ == "__main__":
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)

    args = tyro.cli(main, config=[tyro.conf.ConsolidateSubcommandArgs])
