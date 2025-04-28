# Setups to test the weight optimization procedure under various constraints

import fire
import os
import torch
import matplotlib.pyplot as plt

from datetime import datetime
import json

from objectives import evaluate_mse, bias_regularizer, evaluate_tiv, evaluate_piv
from optimization import optimize_weights_and_privacy_noise
from utils import evaluate_bias_at_clients

SEED = 42

def simple_network_test(
    dimension: int = 128,
    num_trustworthy_ngbrs: int = 1,
    learning_rate: float = 0.5,
    bias_reg: float = 0.5,
    num_iters: int = 2000,
    logfile_path=f"misc/simple_SEED={SEED}.json",
):
    """dimension: Dimension of the vectors whose mean is being estimated
    num_trustworthy_ngbrs: Number of trustworthy neighbors
    learning_rate: Learning rate of the optimization algorithm
    bias_reg: Regularization weight to penalize bias
    num_iters: Number of iterations of the optimization algorithm
    """

    torch.manual_seed(SEED)

    # Data parameters
    radius = 1.0

    # Network connectivity parameters
    num_clients = 10
    pc = 0.9  # Connectivity between clients
    # p = torch.Tensor(
    #     [0.1, 0.1, 0.8, 0.1, 0.1, 0.9, 0.1, 0.1, 0.9, 0.1]
    # )  # Connectivity from clients to PS
    p = torch.Tensor(
        [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    )
    P = pc * torch.ones(num_clients, num_clients)
    P.fill_diagonal_(1)

    # Privacy parameters (Nodes trust their immediate k-hop neighbors)
    delta = 1e-3  # Common delta parameter for differential privacy
    D = delta * torch.ones([num_clients, num_clients])
    eps1 = 1e3  # For oneself and trustworthy neighbors
    eps2 = 1e-2  # For general neighbors
    E = eps2 * torch.ones([num_clients, num_clients])
    for i in range(num_clients):
        E[i][i] = eps1
        for j in range(num_trustworthy_ngbrs):
            E[i][(i + j + 1) % num_clients] = eps1
            E[i][(i + j - 1) % num_clients] = eps1

    # Bias regularization parameters
    reg_type = "L1"
    reg_strength = bias_reg

    weights_opt, noise_opt, losses = optimize_weights_and_privacy_noise(
        p=p,
        P=P,
        E=E,
        D=D,
        radius=radius,
        dimension=dimension,
        reg_type=reg_type,
        reg_strength=reg_strength,
        num_iters=num_iters,
        learning_rate=learning_rate,
    )

    optimized_mse = evaluate_mse(
        p=p, A=weights_opt, P=P, radius=radius, sigma=noise_opt, dimension=dimension
    )
    bias_at_nodes = evaluate_bias_at_clients(p=p, A=weights_opt, P=P)
    total_bias_reg = bias_regularizer(
        p=p, A=weights_opt, P=P, reg_type=reg_type, reg_strength=reg_strength
    )

    # Log results
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging_info = {
        "timestamp": current_time,
        "SEED": SEED,
        "testbench_name": "fully-conn., mediocre conn. to PS, trust k-hop ngbr",
        "radius": radius,
        "dimension": dimension,
        "connectivity_to_PS": p.numpy().tolist(),
        "connectivity_between_clients": pc,
        "privacy_delta": delta,
        "num_trustworthy_ngrs": num_trustworthy_ngbrs,
        "eps_trustworthy": eps1,
        "eps_others": eps2,
        "bias_reg_type": reg_type,
        "bias_reg_strength": reg_strength,
        "num_iters": num_iters,
        "learning_rate": learning_rate,
        "optimized_weights": weights_opt.tolist(),
        "optimized_privacy_noise": noise_opt.tolist(),
        "optimization_loss": [loss.tolist() for loss in losses],
        "optimized_mse": optimized_mse.tolist(),
        "bias_at_nodes": bias_at_nodes.tolist(),
        "total_bias_reg": total_bias_reg.tolist(),
    }

    # Save the logging info to a json file
    if os.path.exists(logfile_path):
        with open(logfile_path, "a") as json_file:
            json.dump(logging_info, json_file, indent=4)
            json_file.write("\n")
    else:
        with open(logfile_path, "w") as json_file:
            json.dump(logging_info, json_file, indent=4)
            json_file.write("\n")

    print(f"Collaboration weights are:\n")
    print(weights_opt)
    print(f"Privacy noises are:\n")
    print(noise_opt)
    print(f"Final objective function value: {losses[-1]}")
    print(f"Optimized MSE: {optimized_mse}")
    print(f"Bias values at nodes: {bias_at_nodes}")
    print(f"Total bias regularization: {total_bias_reg}")

    # Plot the results
    plt.figure(figsize=(14, 7))
    with torch.no_grad():
        plt.plot(losses)

    plt.xlabel("Iterations")
    plt.ylabel("Objective function value")
    plt.yscale("log")
    plt.title("Collaboration weight and privacy noise optimization")
    # plt.savefig(
    #     f"misc/simple_trustworthy_ngbrs_{num_trustworthy_ngbrs}_bias_reg={bias_reg:.0f}_SEED={SEED}_eps2={eps2}_.png"
    # )
    plt.savefig(
        f"misc/sole_good_conn_node_bias_reg={bias_reg:.0f}_SEED={SEED}_eps2={eps2}_pc={pc}_lr={learning_rate}.png"
    )
    # plt.show()

def simple_network_test_seed_variation(
    dimension: int = 128,
    num_trustworthy_ngbrs: int = 1,
    learning_rate: float = 0.01,
    bias_reg: float = 0,
    num_iters: int = 2000,
    logfile_path=f"misc/simple_SEED_variation.json",
):
    """dimension: Dimension of the vectors whose mean is being estimated
    num_trustworthy_ngbrs: Number of trustworthy neighbors
    learning_rate: Learning rate of the optimization algorithm
    bias_reg: Regularization weight to penalize bias
    num_iters: Number of iterations of the optimization algorithm
    """

    SEED_list = [1, 10, 100, 1e3]
    losses_seed_variation = []
    optimized_mse_seed_variation = []
    bias_at_nodes_seed_variation = []
    optimized_tiv_seed_variation = []
    optimized_piv_seed_variation = []

    for SEED in SEED_list:

        print(f"Running with SEED = {SEED}")
        torch.manual_seed(SEED)

        # Data parameters
        radius = 1.0

        # Network connectivity parameters
        num_clients = 10
        pc = 0.9  # Connectivity between clients
        p = torch.Tensor(
            [0.1, 0.1, 0.8, 0.1, 0.1, 0.9, 0.1, 0.1, 0.9, 0.1]
        )  # Connectivity from clients to PS
        P = pc * torch.ones(num_clients, num_clients)
        P.fill_diagonal_(1)

        # Privacy parameters (Nodes trust their immediate k-hop neighbors)
        delta = 1e-3  # Common delta parameter for differential privacy
        D = delta * torch.ones([num_clients, num_clients])
        eps1 = 1e3  # For oneself and trustworthy neighbors
        eps2 = 1  # For general neighbors
        E = eps2 * torch.ones([num_clients, num_clients])
        for i in range(num_clients):
            E[i][i] = eps1
            for j in range(num_trustworthy_ngbrs):
                E[i][(i + j + 1) % num_clients] = eps1
                E[i][(i + j - 1) % num_clients] = eps1

        # Bias regularization parameters
        reg_type = "L1"
        reg_strength = bias_reg

        weights_opt, noise_opt, losses = optimize_weights_and_privacy_noise(
            p=p,
            P=P,
            E=E,
            D=D,
            radius=radius,
            dimension=dimension,
            reg_type=reg_type,
            reg_strength=reg_strength,
            num_iters=num_iters,
            learning_rate=learning_rate,
        )

        optimized_mse = evaluate_mse(
            p=p, A=weights_opt, P=P, radius=radius, sigma=noise_opt, dimension=dimension
        )

        bias_at_nodes = evaluate_bias_at_clients(p=p, A=weights_opt, P=P)
        total_bias_reg = bias_regularizer(
            p=p, A=weights_opt, P=P, reg_type=reg_type, reg_strength=reg_strength
        )

        optimized_tiv = evaluate_tiv(p=p, A=weights_opt, P=P, radius=radius)
        optimized_piv = evaluate_piv(p=p, P=P, sigma=noise_opt, dimension=dimension)

        # Append to lists
        losses_seed_variation.append(losses)
        optimized_mse_seed_variation.append(optimized_mse)
        bias_at_nodes_seed_variation.append(bias_at_nodes)
        optimized_tiv_seed_variation.append(optimized_tiv)
        optimized_piv_seed_variation.append(optimized_piv)

    losses_avg = torch.zeros((len(losses_seed_variation[0]),))
    for i in range(len(losses_seed_variation)):
        losses_avg += torch.Tensor(losses_seed_variation[i])
    losses_avg /= len(losses_seed_variation)

    optimized_mse_avg = torch.sum(torch.Tensor(optimized_mse_seed_variation)) / len(optimized_mse_seed_variation)

    bias_at_nodes_avg = torch.zeros((len(bias_at_nodes_seed_variation[0]),))
    for i in range(len(bias_at_nodes_seed_variation)):
        bias_at_nodes_avg += torch.Tensor(bias_at_nodes_seed_variation[i])
    bias_at_nodes_avg /= len(bias_at_nodes_seed_variation)

    optimized_tiv_avg = torch.sum(torch.Tensor(optimized_tiv_seed_variation)) / len(optimized_tiv_seed_variation)
    optimized_piv_avg = torch.sum(torch.Tensor(optimized_piv_seed_variation)) / len(optimized_piv_seed_variation)

    # Log results
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging_info = {
        "timestamp": current_time,
        "testbench_name": "fully-conn., mediocre conn. to PS, trust k-hop ngbr",
        "radius": radius,
        "dimension": dimension,
        "connectivity_to_PS": p.numpy().tolist(),
        "connectivity_between_clients": pc,
        "privacy_delta": delta,
        "num_trustworthy_ngrs": num_trustworthy_ngbrs,
        "eps_trustworthy": eps1,
        "eps_others": eps2,
        "bias_reg_type": reg_type,
        "bias_reg_strength": reg_strength,
        "num_iters": num_iters,
        "learning_rate": learning_rate,
        "optimization_loss": [loss.tolist() for loss in losses_avg],
        "optimized_mse": optimized_mse_avg.tolist(),
        "bias_at_nodes": bias_at_nodes_avg.tolist(),
        "optimized_weights": weights_opt.tolist(),
        "optimized_privacy_noise": noise_opt.tolist(),
        "optimized_tiv": optimized_tiv_avg.tolist(),
        "optimized_piv": optimized_piv_avg.tolist(),
    }

    # Save the logging info to a json file
    logfile_path = f"misc/simple_trustworthy_ngbrs_{num_trustworthy_ngbrs}_bias_reg={bias_reg:.1f}_lr={1e-2}_pc={pc}_SEED_variation.json"
    if os.path.exists(logfile_path):
        with open(logfile_path, "a") as json_file:
            json.dump(logging_info, json_file, indent=4)
            json_file.write("\n")
    else:
        with open(logfile_path, "w") as json_file:
            json.dump(logging_info, json_file, indent=4)
            json_file.write("\n")

    # Plot the losses average
    plt.figure(figsize=(14, 7))
    with torch.no_grad():
        plt.plot(losses_avg)

    plt.xlabel("Iterations")
    plt.ylabel("Objective function value")
    plt.yscale("log")
    plt.title("Collaboration weight and privacy noise optimization")
    plt.savefig(
        f"misc/simple_trustworthy_ngbrs_{num_trustworthy_ngbrs}_bias_reg={bias_reg:.0f}_lr={1e-2}_pc={pc}_SEED_variation.png"
    )

    print(f"Averaged optimized MSE: {optimized_mse_avg}")
    print(f"Average bias at nodes: {bias_at_nodes_avg}")

    print(f"For the last realization....")
    print(f"Collaboration weights are:\n")
    print(weights_opt)
    print(f"Privacy noises are:\n")
    print(noise_opt)
        
def sole_connected_client_vary_trustworthy_neighbors(
    dimension: int = 128,
    learning_rate: float = 0.5,
    bias_reg: float = 0,
    num_iters: int = 1200,
    logfile_path=f"misc/simple_SEED_variation.json",
):
    """ Vary k in a topology where nodes trust their k hop neighbors and only a single client has good connectivity to the PS
    dimension: Dimension of the vectors whose mean is being estimated
    learning_rate: Learning rate of the optimization algorithm
    bias_reg: Regularization weight to penalize bias
    num_iters: Number of iterations of the optimization algorithm
    """

    SEED_list = [0]
    optimized_mse_seed_variation = []
    optimized_tiv_seed_variation = []
    optimized_piv_seed_variation = []
    losses_vary_num_ngbrs_dict = {}
    biases_vary_num_ngbrs_dict = {}
    weights_opt_num_ngbrs_dict = {}
    noise_opt_num_ngbrs_dict = {}

    # Data parameters
    radius = 1.0

    # Network connectivity parameters
    num_clients = 10
    pc = 0.1  # Connectivity between clients
    print(f"Running with node-node communication probability: {pc}")
    p = torch.Tensor(
        [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    )  # Connectivity from clients to PS -- only one good client
    P = pc * torch.ones(num_clients, num_clients)
    P.fill_diagonal_(1)

    num_trustworthy_ngbrs_list = [i for i in range(num_clients // 2)]

    # Privacy parameters (Nodes trust their immediate k-hop neighbors)
    delta = 1e-3  # Common delta parameter for differential privacy
    D = delta * torch.ones([num_clients, num_clients])
    eps1 = 1e3  # For oneself and trustworthy neighbors
    eps2 = 1e-2  # For general neighbors

    # Bias regularization parameters
    reg_type = "L1"
    reg_strength = bias_reg

    # Make sure you store loss for only one seed
    stored_losses_and_biases = False

    for SEED in SEED_list:

        print(f"Running with SEED = {SEED}")
        torch.manual_seed(SEED)

        optimized_mse_vary_ngbrs = []
        optimized_tiv_vary_num_ngbrs = []
        optimized_piv_vary_num_ngbrs = []

        # Vary number of trustworthy neighbors
        for num_trustworthy_ngbrs in num_trustworthy_ngbrs_list:

            print(f"Number of trustworthy neighbors: {num_trustworthy_ngbrs}")
            print(f"Bias regularization: {bias_reg}")
            print(f"eps_other: {eps2}")

            # Set privacy parameters depending on trustworhty neighbors
            E = eps2 * torch.ones([num_clients, num_clients])
            for i in range(num_clients):
                E[i][i] = eps1
                for j in range(num_trustworthy_ngbrs):
                    E[i][(i + j + 1) % num_clients] = eps1
                    E[i][(i + j - 1) % num_clients] = eps1

            weights_opt, noise_opt, losses = optimize_weights_and_privacy_noise(
                p=p,
                P=P,
                E=E,
                D=D,
                radius=radius,
                dimension=dimension,
                reg_type=reg_type,
                reg_strength=reg_strength,
                num_iters=num_iters,
                learning_rate=learning_rate,
            )

            optimized_mse = evaluate_mse(
                p=p, A=weights_opt, P=P, radius=radius, sigma=noise_opt, dimension=dimension
            )

            bias_at_nodes = evaluate_bias_at_clients(p=p, A=weights_opt, P=P)

            optimized_tiv = evaluate_tiv(p=p, A=weights_opt, P=P, radius=radius)
            optimized_piv = evaluate_piv(p=p, P=P, sigma=noise_opt, dimension=dimension)

            # Add to list
            optimized_mse_vary_ngbrs.append(optimized_mse)
            optimized_tiv_vary_num_ngbrs.append(optimized_tiv)
            optimized_piv_vary_num_ngbrs.append(optimized_piv)

            # Store losses only for the first seed
            if not stored_losses_and_biases:
                losses_vary_num_ngbrs_dict[str(num_trustworthy_ngbrs)] = losses
                biases_vary_num_ngbrs_dict[str(num_trustworthy_ngbrs)] = bias_at_nodes.tolist()
                weights_opt_num_ngbrs_dict[str(num_trustworthy_ngbrs)] = weights_opt.tolist()
                noise_opt_num_ngbrs_dict[str(num_trustworthy_ngbrs)] = noise_opt.tolist()

        stored_losses_and_biases = True

        optimized_mse_seed_variation.append(optimized_mse_vary_ngbrs)
        optimized_tiv_seed_variation.append(optimized_tiv_vary_num_ngbrs)
        optimized_piv_seed_variation.append(optimized_piv_vary_num_ngbrs)

    optimized_mse_vary_ngbrs_avg = torch.zeros((len(optimized_mse_seed_variation[0]),))
    for i in range(len(optimized_mse_seed_variation)):
        optimized_mse_vary_ngbrs_avg += torch.Tensor(optimized_mse_seed_variation[i])
    optimized_mse_vary_ngbrs_avg /= len(optimized_mse_seed_variation)

    optimized_tiv_vary_ngbrs_avg = torch.zeros((len(optimized_tiv_seed_variation[0]),))
    for i in range(len(optimized_tiv_seed_variation)):
        optimized_tiv_vary_ngbrs_avg += torch.Tensor(optimized_tiv_seed_variation[i])
    optimized_tiv_vary_ngbrs_avg /= len(optimized_tiv_seed_variation)

    optimized_piv_vary_ngbrs_avg = torch.zeros((len(optimized_piv_seed_variation[0]),))
    for i in range(len(optimized_piv_seed_variation)):
        optimized_piv_vary_ngbrs_avg += torch.Tensor(optimized_piv_seed_variation[i])
    optimized_piv_vary_ngbrs_avg /= len(optimized_piv_seed_variation)

    # Log results
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging_info = {
        "timestamp": current_time,
        "testbench_name": "fully-conn., mediocre conn. to PS, trust k-hop ngbr",
        "radius": radius,
        "dimension": dimension,
        "connectivity_to_PS": p.numpy().tolist(),
        "connectivity_between_clients": pc,
        "privacy_delta": delta,
        "num_trustworthy_ngrs": num_trustworthy_ngbrs,
        "eps_trustworthy": eps1,
        "eps_others": eps2,
        "bias_reg_type": reg_type,
        "bias_reg_strength": reg_strength,
        "num_iters": num_iters,
        "learning_rate": learning_rate,
        "optimized_mse_vary_ngbrs": optimized_mse_vary_ngbrs_avg.tolist(),
        "optimized_tiv_vary_k": optimized_tiv_vary_ngbrs_avg.tolist(),
        "optimized_piv_vary_k": optimized_piv_vary_ngbrs_avg.tolist(),
        "bias_at_nodes_vary_k": biases_vary_num_ngbrs_dict,
        "weights_opt_vary_k": weights_opt_num_ngbrs_dict,
        "noise_opt_vary_k": noise_opt_num_ngbrs_dict,
    }

    # Save the logging info to a json file
    logfile_path = f"misc/vary_trustworthy_ngbrs_bias_reg={bias_reg:.1f}_lr={learning_rate}_pc={pc}_eps2={eps2}.json"
    if os.path.exists(logfile_path):
        with open(logfile_path, "a") as json_file:
            json.dump(logging_info, json_file, indent=4)
            json_file.write("\n")
    else:
        with open(logfile_path, "w") as json_file:
            json.dump(logging_info, json_file, indent=4)
            json_file.write("\n")

    print(f"Averaged optimized MSE with varying number of trustworthy neighbors: {optimized_mse_vary_ngbrs}")

def simple_network_test_vary_num_trustworthy_ngbrs():
    """Vary the number of trustworthy neighbors and record values"""

    # trustworthy_ngbrs_list = [1, 2, 3, 4, 5]
    # trustworthy_ngbrs_list = [1]
    trustworthy_ngbrs_list = [0]
    dimension = 1
    bias_reg = 1e2
    learning_rate = 1e-2
    num_iters = 3000

    for num_trustworthy_ngbrs in trustworthy_ngbrs_list:
        simple_network_test(
            dimension=dimension,
            num_trustworthy_ngbrs=num_trustworthy_ngbrs,
            bias_reg=bias_reg,
            learning_rate=learning_rate,
            num_iters=num_iters,
            logfile_path=f"misc/simple_vary_trustworthy_ngrs_dim=1_bias_reg={bias_reg:.0f}_lr={learning_rate}_iters={num_iters}_SEED={SEED}.json",
        )


if __name__ == "__main__":

    # Experiment with sole good client: MSE values for different eps_other. Tuning learning rate
    # print(f"Expt: Sole good client")
    # simple_network_test()

    # Experiment with variation of number of trustworthy neighbors
    print(f"Expt: Variation with number of trusthworthy neighbors")
    sole_connected_client_vary_trustworthy_neighbors(bias_reg=0.5)

    # Experimenting with variation of bias regularization parameter
    #  bias_reg_arr = [1e3]

    # for bias_reg in bias_reg_arr:
    #     print(f"Running with bias regularization parameter: {bias_reg}")
    #     simple_network_test_seed_variation(bias_reg=bias_reg)

    # fire.Fire(simple_network_test_vary_num_trustworthy_ngbrs)

    # fire.Fire(simple_network_test)
