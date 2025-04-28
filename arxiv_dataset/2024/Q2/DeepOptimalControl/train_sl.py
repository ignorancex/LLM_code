import os
import time
import scipy.io
import numpy as np

from modules.supervise import ModelControl
from utils.base import set_seed_everywhere, parse_args


args = parse_args()

print(args.experiment_name)
if (args.experiment_name == "quadrotor_fix") or (args.experiment_name == "quadrotor_adaptive"):
    import problems.quadrotor_fix as example_fix
    try:
        space, tT = args.problem_id.split("_")
        print("Space and Time:", space, tT)
        problem_fix = example_fix.ProblemSetup(
            float(space), float(tT), args.device)
    except:
        raise "Not right problem id."
elif args.experiment_name == "satellite_fix":
    import problems.satellite_fix as example_fix
    problem_fix = example_fix.ProblemSetup()
else:
    raise "Not implemented problem."

args.experiment_name = args.problem_id + "_" + args.experiment_name
args.data_train = "./data/" + args.experiment_name + '/' + args.data_train
args.scaling = "./data/" + args.experiment_name + '/' + args.scaling
args.data_valid = "./data/" + args.experiment_name + '/' + args.data_valid

problem_fix.device = args.device
problem_fix.batch_size = args.batch_size
problem_fix.learning_rate = args.lr

def run():
    # Load data
    train_data = scipy.io.loadmat(args.data_train)
    valid_data = scipy.io.loadmat(args.data_valid)
    scaling = scipy.io.loadmat(args.scaling)

    N_train = train_data['X'].shape[1]
    N_val = valid_data['X'].shape[1]

    if 'U' not in train_data.keys():
        for data in [train_data, valid_data]:
            data['U'] = problem_fix.U_star(np.vstack((data['X'], data['A'])))
        scaling['U_lb'] = train_data['U'].min()
        scaling['U_ub'] = train_data['U'].max()

    print('\n Number of training data:', N_train)
    print('\n Number of validation data:', N_val)

    save_path = './output/%s/' % args.experiment_name
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        save_path = './output/%s/model/' % args.experiment_name
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    # Set model
    demo = ModelControl(problem_fix)
    demo.Scaling(scaling)

    if args.load_model:
        load_path_tf = f'./output/{args.experiment_name}/model/supervised_{demo.model_u.__class__.__name__}.pkl'
        print(f"Load model from {load_path_tf}.")
        demo.load_model(load_path_tf)

    print('Start training.')
    start_time = time.time()
    demo.train_model(train_data, valid_data,
                     args.num_epochs, args.experiment_name)
    train_time = time.time() - start_time
    print('Training time is: %.0f' % (train_time), 'sec.')


if __name__ == "__main__":
    set_seed_everywhere(0)
    run()
