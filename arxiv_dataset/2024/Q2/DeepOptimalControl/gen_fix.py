import os
import numpy as np
import scipy.io
import time
import timeit
import functools
import multiprocessing
import utils.bvp_solvers as bvp_solvers
import utils.X0_samplers as X0_samplers
import problems.quadrotor_fix as example_fix
from utils.base import set_seed_everywhere, parse_args


def data_parse(data_free_dict):
    X = data_free_dict['X']
    A = data_free_dict['A']
    U = data_free_dict['U']
    V = data_free_dict['V']
    t = data_free_dict['t'].reshape((1, -1))

    print('The shape of X is:{}'.format(X.shape))
    print('The shape of A is:{}'.format(A.shape))
    print('The shape of U is:{}'.format(U.shape))
    print('The shape of V is:{}'.format(V.shape))
    print('The shape of t is:{}'.format(t.shape))

    data = {'X': X,
            'A': A,
            'U': U,
            'V': V,
            't': t}

    if args.data_type == 'train':
        scaling = {
            'lb': np.min(X, axis=1, keepdims=True),
            'ub': np.max(X, axis=1, keepdims=True),
            'A_lb': np.min(A, axis=1, keepdims=True),
            'A_ub': np.max(A, axis=1, keepdims=True),
            'U_lb': np.min(U, axis=1, keepdims=True),
            'U_ub': np.max(U, axis=1, keepdims=True),
            'V_min': np.min(V).flatten(),
            'V_max': np.max(V).flatten()}
        scipy.io.savemat(save_path + 'data_fix_scaling.mat', scaling)
        scipy.io.savemat(save_path + 'data_fix_train.mat', data)
    else:
        scipy.io.savemat(save_path + 'data_fix_%s.mat' %
                         args.data_type, data)


def solve_problem_single():
    N_states = problem.N_states
    X0_sampler = X0_samplers.uniform
    X0s = X0_sampler(args.data_size, N_states,
                        problem.X0_lb, problem.X0_ub)

    Nums = X0s.shape[1]
    print('Total Nums of trajectories:{}'.format(Nums))
    print('*************Warm by fix {}****************'.format(args.bvp_type))
    time_yes_list = []
    time_no_list = []
    t_list = []
    X_list = []
    A_list = []
    U_list = []
    V_list = []

    # Solves the TPBVP with fix-time-problem
    k = 0
    fail_fix_index = []
    if args.bvp_type == 'no_march':
        sol_fix = bvp_solver_fix_direct(problem=problem, x0=X0)
    elif args.bvp_type == 'space_march_fw':
        sol_fix = bvp_solver_fix_march_fw(problem=problem, k_try=args.k_try, x0=X0)
    else:
        raise "Not implemented bvp type."

    while (k < Nums):
        time_start = time.time()
        print('*************Solving BVP #{} *************'.format(k+1))
        X0 = X0s[:, k:k+1]

        if not sol_fix.success:
            print('&&&&&&&&&&&&& The sol_fix solver failed ! &&&&&&&&&&&')
            fail_fix_index.append(k)
            time_no_list.append(time.time() - time_start)
            X0s[:, k:k+1] = X0_sampler(1, N_states, problem.X0_lb, problem.X0_ub)
        else:
            print('The problem has been solved!')
            t_fix = np.linspace(0, problem.tT, 100)
            X_aug_BVP = sol_fix.sol(t_fix)
            U_BVP = problem.U_star(X_aug_BVP)
            V_BVP = problem.compute_cost(t_fix, X_aug_BVP[:N_states], U_BVP)
            t_list.append(t_fix)
            X_list.append(X_aug_BVP[:N_states, :])
            A_list.append(X_aug_BVP[N_states:, :])
            U_list.append(U_BVP)
            V_list.append(V_BVP)
            time_yes_list.append(time.time()-time_start)
            k += 1

    succ_num = len(time_yes_list)
    fail_num = len(time_no_list)
    f = open("out.txt", "a")
    print('='*50, file=f)
    print(f"Time cost for free_end_time case: {sum(time_yes_list)+sum(time_no_list)}", file=f)
    print(f"The success rate is: {succ_num}/{fail_num+succ_num}; Average success time is: {sum(time_yes_list)/(1e-8+succ_num)}.", file=f)
    print(f"The fail rate is:{fail_num}/{fail_num+succ_num}; Average fail time is:{sum(time_no_list)/(1e-8+fail_num)}", file=f)
    print(f"fail_fix_index: {fail_fix_index}", file=f)
    f.close()

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data_fix_dict = {}
    data_fix_dict['t'] = np.hstack(t_list)
    data_fix_dict['X'] = np.hstack(X_list)
    data_fix_dict['A'] = np.hstack(A_list)
    data_fix_dict['U'] = np.hstack(U_list)
    data_fix_dict['V'] = np.hstack(V_list)
    data_parse(data_fix_dict)
    return


def solve_one(problem, k_try, N_states, data):
    i, X0 = data
    print(f'*************Solving BVP #{i+1} *************')
    if args.bvp_type=='no_march':
        sol_fix = bvp_solver_fix_direct(problem=problem, x0=X0)
    elif args.bvp_type=='space_march_fw':
        sol_fix = bvp_solver_fix_march_fw(problem=problem, k_try=k_try, x0=X0)
    else:
        raise "None Implemented bvp type."
        
    if not sol_fix.success:
        print('&&&&&&&&&&&&& The sol_fix solver failed ! &&&&&&&&&&&')
        return False, 0, 0, 0, 0
    else:
        print('The problem has been solved!')
        t_fix = np.linspace(0, problem.tT, 100)
        X_aug_BVP = sol_fix.sol(t_fix)
        U_BVP = problem.U_star(X_aug_BVP)
        V_BVP = problem.compute_cost(t_fix, X_aug_BVP[:N_states], U_BVP)  # tf=None for fix problem
        return True, t_fix, X_aug_BVP, U_BVP, V_BVP


def solve_problem_multi():
    N_states = problem.N_states
    X0_sampler = X0_samplers.uniform
    X0s = X0_sampler(args.data_size, N_states, problem.X0_lb, problem.X0_ub)

    Nums = X0s.shape[1]
    print('Total nums of trajectories: {}'.format(Nums))
    print('*************Warm by fix {}****************'.format(args.bvp_type))
    
    # Check num of cores
    num_cores = int(multiprocessing.cpu_count())
    print(f"There are {num_cores} CPU cores. Using {args.num_processors} processors.")

    # Solves the TPBVP with fix-time-problem 
    init_states = [(k, X0s[:,k:k+1]) for k in range(Nums)]
    p = multiprocessing.Pool(args.num_processors)
    start = timeit.default_timer()
    results = p.map(functools.partial(solve_one, problem, args.k_try, N_states), init_states)
    p.close()
    p.join()
    end = timeit.default_timer()
    print('Multi processing time:', str(end-start), 's')

    t_list = [x[1] for x in results if x[0]]
    X_list = [x[2][:N_states,:] for x in results if x[0]]
    A_list = [x[2][N_states:,:] for x in results if x[0]]
    U_list = [x[3] for x in results if x[0]]
    V_list = [x[4] for x in results if x[0]]
    
    succ_num = sum([x[0] for x in results])
    fail_num = len(results) - succ_num
    
    f = open('out.txt','a')
    print('='*50, file=f)
    print('Time cost: {}'.format(end-start), file=f)
    print('The success rate is:{}/{}; Total time is:{}, Average success time is:{}.'.format(
        succ_num, fail_num+succ_num, end-start, (end-start)/(1e-8+succ_num)), file=f)
    f.close()

    if not os.path.exists(args.save_path):
      os.makedirs(args.save_path) 

    data_fix_dict = {}
    data_fix_dict['t'] = np.hstack(t_list)
    data_fix_dict['X'] = np.hstack(X_list)
    data_fix_dict['A'] = np.hstack(A_list)
    data_fix_dict['U'] = np.hstack(U_list)
    data_fix_dict['V'] = np.hstack(V_list)
    data_parse(data_fix_dict)
    return


if __name__ == "__main__":
    bvp_solver_fix_direct = bvp_solvers.direct_fix
    bvp_solver_fix_march_fw = bvp_solvers.space_march_fix_forward

    args = parse_args()
    save_path = './data/' + args.problem_id + "_" + args.experiment_name + '/'

    if args.experiment_name == "quadrotor_fix":
        import problems.quadrotor_fix as example_fix
        try:
            space, tT = args.problem_id.split("_")
            print("Space and Time:", space, tT)
            problem = example_fix.ProblemSetup(float(space), float(tT))
        except:
            raise "Invalid problem id."
    else:
        raise "Not implemented problem."

    set_seed_everywhere(args.seed)
    if args.num_processors == 1:
        result = solve_problem_single()
    elif args.num_processors > 1:
        result = solve_problem_multi()
    else:
        raise "Invalid number of processors."
