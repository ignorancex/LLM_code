
import argparse
import os
import torch
import scipy.io
import numpy as np
from tqdm import tqdm
from scipy.integrate import solve_ivp

from modules.supervise import ModelControl
from modules.networks import DNN_U
from utils.base import set_seed_everywhere
from utils.plot import plot_cmp, plot_cmp_sto
import utils.bvp_solvers as bvp_solvers
bvp_solver_fix_march_fw = bvp_solvers.space_march_fix_forward
bvp_solver_fix_direct = bvp_solvers.direct_fix


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--problem_id", default="", type=str)
    parser.add_argument("--experiment_name", default="", type=str)
    parser.add_argument("--direct_iter", default="", type=str)
    parser.add_argument("--sl_iter", default="", type=str)
    parser.add_argument("--finetune_iter", default="", type=str)

    args = parser.parse_args()
    return args


args = parse_args()
problem_id = args.problem_id
experiment_name = args.experiment_name
direct_iter = args.direct_iter
sl_iter = args.sl_iter
finetune_iter = args.finetune_iter
experiment = experiment_name.split("_")[0]
if experiment == "quadrotor":
    import problems.quadrotor_fix as example_fix
    try:
        space, tT = problem_id.split("_")
        print("Space and Time:", space, tT)
        problem = example_fix.ProblemSetup(float(space), float(tT))
    except:
        raise "Not right problem id."
elif experiment == "satellite":
    import problems.satellite_fix as example_fix
    space, tT = problem_id.split("_")
    problem = example_fix.ProblemSetup()
    problem.U_d = 0.
else:
    raise "Not implemented problem."


scaling = f"data/{problem_id}_{experiment_name}/data_fix_scaling.mat"
valid_data = f"data/{problem_id}_{experiment_name}/data_fix_valid.mat"
load_model_path = f"output/{problem_id}_{experiment_name}/model/"


# Loads pre-trained NN for Indirect control
scaling = scipy.io.loadmat(scaling)
lb = torch.tensor(scaling['lb'].T.astype(np.float32))
ub = torch.tensor(scaling['ub'].T.astype(np.float32))

valid_data = scipy.io.loadmat(valid_data)
idx0 = np.nonzero(np.equal(valid_data.pop('t'), 0.))[1]
valid_data_init = valid_data['X'][:, idx0]
valid_data_cost = valid_data['V'][:, idx0]


# Initializes some parameters
hidden_size = 64
N_states = problem.N_states
N_controls = problem.N_controls

indirect_model = ModelControl(problem)
indirect_model.Scaling(scaling)
indirect_model.model_u.load_state_dict(torch.load(
    load_model_path + f"supervised_DNN_U_{sl_iter}.pkl", map_location="cpu"))

layers = [N_states+1, hidden_size, hidden_size, hidden_size, N_controls]
node_model = DNN_U(layers)
node_model.load_state_dict(torch.load(
    load_model_path + f"direct_DNN_U_{direct_iter}.pkl", map_location="cpu"))

finetune_model = DNN_U(layers)
finetune_model.load_state_dict(torch.load(load_model_path + f"finetune_DNN_U_{sl_iter}_{finetune_iter}.pkl", map_location="cpu"))



def finetune_eval_U(t, x):
    t=torch.tensor(t).float().T
    x=torch.tensor(x).float().T
    t=2. * t / problem.tT - 1.
    x=2. * (x - lb) / (ub - lb) - 1.
    out=finetune_model(t, x)
    return out.detach().numpy().T + problem.U_d


def node_eval_U(t, x):
    t=torch.tensor(t).float().T
    x=torch.tensor(x).float().T
    out=node_model(t, x)
    return out.detach().numpy().T + problem.U_d

def get_stat(alg_cost_list, BVP_cost_list, env = "deter", alg = "Unknown", sigma = 0.):
    assert len(alg_cost_list) == len(BVP_cost_list)
    print(f"{alg} COST-RATE STAT")
    rate_list=[alg_cost_list[i]/BVP_cost_list[i][0]
        for i in range(len(alg_cost_list))]
    print("Mean | Std | Max | Min | 90-percentile | 75-percentile | Median")
    print(np.mean(rate_list),
        np.std(rate_list),
        np.max(rate_list),
        np.min(rate_list),
        np.percentile(rate_list, 90),
        np.percentile(rate_list, 75),
        np.median(rate_list))

    save_path=f"output/{problem_id}_{experiment_name}/results"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if env == "deter":
        np.savetxt(f"{save_path}/{env}_{alg}.txt", alg_cost_list)
    else:
        np.savetxt(f"{save_path}/{env}_{alg}_{sigma}.txt", alg_cost_list)
    return rate_list


def compare_deterministic():
    X0s=valid_data_init
    print(X0s.shape)

    DO_cost_list=[]
    SL_cost_list=[]
    FINETUNE_cost_list=[]
    BVP_cost_list=[]

    for i in tqdm(range(X0s.shape[1])):
        X0=X0s[:, i]
        BVP_cost=valid_data_cost[:, i]

        # Direct Controller
        SOL=solve_ivp(problem.dynamics, [0., tT], X0,
                        method = "RK23", args = (node_eval_U,), rtol = 1e-05, atol = 1e-05)
        U_NN=node_eval_U(SOL.t.reshape(1, -1), SOL.y)
        save_dict={'t_DO': SOL.t, 'X_DO': SOL.y, 'U_DO': U_NN}
        print(save_dict['X_DO'].shape, save_dict["U_DO"].shape)
        print("="*30)

        # Fine-tuned Controller
        SOL=solve_ivp(problem.dynamics, [0., tT], X0,
                        method = "RK23", args = (finetune_eval_U,), rtol = 1e-05, atol = 1e-05)
        U_NN=finetune_eval_U(SOL.t.reshape(1, -1), SOL.y)
        save_dict.update(
            {'t_FINETUNE': SOL.t, 'X_FINETUNE': SOL.y, 'U_FINETUNE': U_NN})
        print(save_dict['X_FINETUNE'].shape, save_dict["U_FINETUNE"].shape)
        print("="*30)

        # SL Controller
        SOL= solve_ivp(problem.dynamics, [0., tT], X0,
                        method ="RK23", args=(indirect_model.eval_U,), rtol=1e-05, atol=1e-05)
        print(SOL.t.shape, SOL.y.shape)
        U_NN = indirect_model.eval_U(SOL.t.reshape(1, -1), SOL.y)
        save_dict.update({'t_SL': SOL.t, 'X_SL': SOL.y, 'U_SL': U_NN})
        print(save_dict['X_SL'].shape, save_dict["U_SL"].shape)

        DO_cost= problem.compute_cost(
            save_dict['t_DO'], save_dict['X_DO'], save_dict['U_DO'])[0,0]
        FINETUNE_cost = problem.compute_cost(
            save_dict['t_FINETUNE'], save_dict['X_FINETUNE'], save_dict['U_FINETUNE'])[0,0]
        SL_cost = problem.compute_cost(
            save_dict['t_SL'], save_dict['X_SL'], save_dict['U_SL'])[0,0]

        print('DO cost: %.2f' % (DO_cost))
        print('SL cost: %.2f' % (SL_cost))
        print('SL+DO cost: %.2f' % (FINETUNE_cost))
        print('BVP cost: %.2f' % (BVP_cost))
        print('='*20)
        DO_cost_list.append(DO_cost)
        FINETUNE_cost_list.append(FINETUNE_cost)
        SL_cost_list.append(SL_cost)
        BVP_cost_list.append(BVP_cost)
    print("DO | SL | SL+DO | BVP")
    print("Mean:", np.mean(DO_cost_list), np.mean(SL_cost_list), np.mean(FINETUNE_cost_list), np.mean(BVP_cost_list))
    print("Std:", np.std(DO_cost_list), np.std(SL_cost_list), np.std(FINETUNE_cost_list), np.std(BVP_cost_list))

    print("="*20)
    DO_rate_list = get_stat(DO_cost_list, BVP_cost_list, alg="DO")
    print("="*20)
    SL_rate_list = get_stat(SL_cost_list, BVP_cost_list, alg="SL")
    print("="*20)
    FINETUNE_rate_list = get_stat(FINETUNE_cost_list, BVP_cost_list, alg="SL+DO")
    print(DO_rate_list)
    results = {"Pre-train and Fine-tune": FINETUNE_rate_list, "Supervised Learning": SL_rate_list, "Direct Optimization": DO_rate_list}
    plot_cmp(results, problem_id, experiment_name)
    return 



def noisy_results(X0, sigma):
    # Initializes some parameters
    dt = 1e-02
    t = np.arange(0., problem.tT+dt/2., dt)
    Nt = t.shape[0]
    def RK4_step(f, t, X, U_fun, W):
        '''Simulates ZOH with measurement noise.'''
        U_eval = U_fun(t.reshape(1,1)+W[0], np.reshape(X+W[1:], (-1,1)))
        U = lambda t, X: U_eval

        # M = 4 steps of RK4 for each sample taken
        M = 4
        _dt = dt/M
        t0 = np.copy(t)
        X1 = np.copy(X)
        for _ in range(M):
            k1 = _dt * f(t, X, U)
            k2 = _dt * f(t + _dt/2., X + k1/2., U)
            k3 = _dt * f(t + _dt/2., X + k2/2., U)
            k4 = _dt * f(t + _dt, X + k3, U)

            X1 += (k1 + 2.*(k2 + k3) + k4)/6.
            t0 += _dt

        return X1

    W = sigma * (2 * np.random.rand(N_states+1, Nt) - 1)

    X_DO = np.empty((N_states, Nt))
    X_SL = np.empty((N_states, Nt))
    X_FINETUNE = np.empty((N_states, Nt))
    X_DO[:, 0] = X0
    X_SL[:, 0] = X0
    X_FINETUNE[:, 0] = X0

    # Integrates the closed-loop system (NN & LQR controllers, RK4)
    for k in range(1,Nt):
        X_DO[:,k] = RK4_step(problem.dynamics, t[k-1], X_DO[:,k-1], node_eval_U, W[:,k-1])
        X_SL[:,k] = RK4_step(problem.dynamics, t[k-1], X_SL[:,k-1], indirect_model.eval_U, W[:,k-1])
        X_FINETUNE[:,k] = RK4_step(problem.dynamics, t[k-1], X_FINETUNE[:,k-1], finetune_eval_U, W[:,k-1])

    U_DO = node_eval_U(t.reshape(1,-1)+W[0:1, :], X_DO + W[1:, :])
    U_SL = indirect_model.eval_U(t.reshape(1,-1)+W[0:1, :], X_SL + W[1:, :])
    U_FINETUNE = finetune_eval_U(t.reshape(1,-1)+W[0:1, :], X_FINETUNE + W[1:, :])
    
    save_dict = {'t': t, 
                'X_DO': X_DO, 'U_DO': U_DO, 
                'X_SL': X_SL, 'U_SL': U_SL, 
                'X_FINETUNE': X_FINETUNE, 'U_FINETUNE': U_FINETUNE,
                }

    DO_cost = problem.compute_cost(
        save_dict['t'], save_dict['X_DO'], save_dict['U_DO'])[0,0]
    SL_cost = problem.compute_cost(
        save_dict['t'], save_dict['X_SL'], save_dict['U_SL'])[0,0]
    FINETUNE_cost = problem.compute_cost(
        save_dict['t'], save_dict['X_FINETUNE'], save_dict['U_FINETUNE'])[0,0]

    print('DO cost: %.2f' % (DO_cost))
    print('SL cost: %.2f' % (SL_cost))
    print('SL+DO cost: %.2f' % (FINETUNE_cost))
    print('='*20)
    return DO_cost, SL_cost, FINETUNE_cost



def compare_stochastic(sigma=0.0314):
    N_init = valid_data_init.shape[1]
    assert N_init >= 2, "Num of init states must be larger than 2."
    DO_cost_list = np.zeros(N_init)
    SL_cost_list = np.zeros(N_init)
    FINETUNE_cost_list = np.zeros(N_init)
    BVP_cost_list = valid_data_cost.reshape(-1, 1)

    set_seed_everywhere(0)
    for i in range(N_init):
        DO_cost, SL_cost, FINETUNE_cost = noisy_results(valid_data_init[:, i], sigma)
        DO_cost_list[i] = DO_cost
        SL_cost_list[i] = SL_cost
        FINETUNE_cost_list[i] = FINETUNE_cost
        print("BVP:", valid_data_cost[:, i])

    print("DO | SL | SL+DO | BVP")
    print("Mean:", np.mean(DO_cost_list), np.mean(SL_cost_list), np.mean(FINETUNE_cost_list), np.mean(BVP_cost_list))
    print("Std:", np.std(DO_cost_list), np.std(SL_cost_list), np.std(FINETUNE_cost_list), np.std(BVP_cost_list))

    DO_rate_list = get_stat(DO_cost_list, BVP_cost_list, env="sto", alg="DO", sigma=sigma)
    SL_rate_list = get_stat(SL_cost_list, BVP_cost_list, env="sto", alg="SL", sigma=sigma)
    FINETUNE_rate_list = get_stat(FINETUNE_cost_list, BVP_cost_list, env="sto", alg="SL+DO", sigma=sigma)
    results = {"Pre-train and Fine-tune": FINETUNE_rate_list, "Supervised Learning": SL_rate_list, "Direct Optimization": DO_rate_list}
    plot_cmp_sto(results, problem_id, experiment_name, sigma)



if __name__ == "__main__":
    set_seed_everywhere(177)
    compare_deterministic()
    sigma_list = [0.01, 0.025, 0.05]
    for sigma in sigma_list:
        compare_stochastic(sigma)



