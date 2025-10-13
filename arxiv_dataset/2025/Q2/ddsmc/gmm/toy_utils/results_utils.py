import torch
import os
import csv
import matplotlib.pyplot as plt

def get_file_name(args, extension):
    if args.algo.lower() == "mcgdiff":
        base = f"{args.algo}_xdim={args.dim_x}_ydim={args.dim_y}_seed={args.seed}_nparticles={args.num_particles}_nsteps={args.num_steps}_kappa={args.kappa}"
    elif args.algo.lower() == "ddsmc":
        base = f"{args.algo}_xdim={args.dim_x}_ydim={args.dim_y}_seed={args.seed}_nparticles={args.num_particles}_nsteps={args.num_steps}_eta={args.eta_ddsmc}_ode={args.use_ode}"
        if args.max_num_ode_steps is not None:
            base += f"_max_odesteps={args.max_num_ode_steps}"
        else:
            base += f"_max_odesteps={args.num_steps}"
    elif args.algo.lower() == "tds":
        base = f"{args.algo}_xdim={args.dim_x}_ydim={args.dim_y}_seed={args.seed}_nparticles={args.num_particles}_nsteps={args.num_steps}"
    elif args.algo.lower() == "dcps":
        base = f"{args.algo}_xdim={args.dim_x}_ydim={args.dim_y}_seed={args.seed}_nsteps={args.num_steps}_M={args.dcps_M}"
    elif args.algo.lower() == "daps":
        base = f"{args.algo}_xdim={args.dim_x}_ydim={args.dim_y}_seed={args.seed}_nsteps={args.num_steps}"
        if args.max_num_ode_steps is not None:
            base += f"_max_odesteps={args.max_num_ode_steps}"
        else:
            base += f"_max_odesteps={args.num_steps}"
    else:
        base = f"{args.algo}_xdim={args.dim_x}_ydim={args.dim_y}_seed={args.seed}_nsteps={args.num_steps}"
    return f"{base}.{extension}"


def save_swd_to_file(swd, loglik, runtime, args):
    if not args.num_samples == 10000 or args.debug:
        return
    if not os.path.isdir("toy_results"):
        os.mkdir("toy_results")
    folder = f"xdim={args.dim_x}_ydim={args.dim_y}"
    folder = os.path.join("toy_results", folder)
    if not os.path.isdir(folder):
        os.mkdir(folder)
    filename = get_file_name(args, "csv")
    file_path = os.path.join(folder, filename)
    if not os.path.isfile(file_path):
        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["seed", "swd", "loglik", "runtime"])
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([args.seed, swd, loglik, runtime])


def split_number(number_to_split, max_value):
    q, r = divmod(number_to_split, max_value)
    result = [max_value] * q
    if r > 0:
        result.append(r)
    return result

def plot_samples(samples, reference_samples, args):
    if args.debug:
        if not os.path.isdir("toy_figures_debug"):
            os.mkdir("toy_figures_debug")
        folder = os.path.join("toy_figures_debug", f"xdim={args.dim_x}_ydim={args.dim_y}")
    else:
        if not os.path.isdir("toy_figures"):
            os.mkdir("toy_figures")
        folder = os.path.join("toy_figures", f"xdim={args.dim_x}_ydim={args.dim_y}_numsamples={args.num_samples}")
    if not os.path.isdir(folder):
        os.mkdir(folder)
    marker_size=10
    plt_alpha = 0.5
    plt.scatter(*reference_samples[:, :2].T, label="Posterior", alpha=plt_alpha, s=marker_size)
    plt.scatter(*samples[:, :2].T, label=args.algo, alpha=plt_alpha, s=marker_size)
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)
    plt.legend()
    filepath = os.path.join(folder, get_file_name(args, "png"))
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()

def save_generated_samples(samples, reference_samples, args):
    if not args.num_samples == 10000 or args.debug:
        return
    if not os.path.isdir("toy_samples"):
        os.mkdir("toy_samples")
    folder = os.path.join("toy_samples", f"xdim={args.dim_x}_ydim={args.dim_y}")
    if not os.path.isdir(folder):
        os.mkdir(folder)

    ref_file = f"ref_xdim={args.dim_x}_ydim={args.dim_y}_seed={args.seed}.pt"
    torch.save(reference_samples, os.path.join(folder, ref_file))

    filename = get_file_name(args, "pt")
    torch.save(samples, os.path.join(folder, filename))