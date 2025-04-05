import os
import shelve
import sys
import os
import pandas as pd

print(os.path.abspath(os.path.dirname(__file__)))

from experiments.base.experiments_base import *
from experiments.base.set_data import *
from src.evaluation.evaluate_base import *
from src.publish_evals.publish_evals import *

from datetime import datetime
import time
import os
import argparse
import config as config

#acknowledgement: this file is partially referenced from our baseline model picker. We extended it to be suitable for online contextual data streaming settings.

def run_experiment(dataset, stream_size, stream_setting, budgets, num_reals, which_methods, policy_index=-1,random_policy=False):
    """
    The main script for running online model selection experiments.
    """

    start_time = time.time()

    """Create a results directory."""
    now = datetime.now().strftime("_Date-%Y-%m-%d_Time-%H-%M") # get datetime
    which_methods_print = ''.join(map(str, which_methods))

    print("              dataset:",dataset)
    print("          stream_size:",stream_size)
    print("       stream_setting:",stream_setting)
    print("              budgets:",budgets)
    print("            num_reals:",num_reals)
    print("        which_methods:",which_methods)
    print("         policy_index:",policy_index)
    print("        random_policy:",random_policy)
    print("                 task:",config.task)

    folder_= dataset+'_streamsize'+str(stream_size)+'_numreals'+str(num_reals)+str(now)+'_which_methods'+str(which_methods_print)+"_policy"+str(policy_index)
    path_='resources/results/'+folder_
    print(path_)

    os.mkdir(path_) # create the folder
    results_dir = Path(path_) # assign it to the results directory var

    """Set data."""
    # Set the data
    data = SetData(dataset, stream_size, stream_setting, budgets, num_reals, results_dir, which_methods,policy_index,random_policy) # data class
    # Save
    data.save_data()


    """Run experiments."""
    print("\n# Running Experiments\n")
    experiments_base(data, cache=None)

    """Evaluate data."""
    print("\n# Running Final Evaluation\n")
    Evals(data)

    """Announce the evaluation results."""
    publish_evals(results_dir)
    elapsed_time = time.time() - start_time

    print("\n# Work Completed...\n")
    print("Elapsed time: %s" % time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    print("The experiment results can be found at: %s" % str(results_dir))
    print("foldername")
    print(folder_)
    print("Time")
    print(now)


if __name__ == "__main__":

    np.random.seed(30)
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-stream_setting',default="floating")


    #  which_methods = ['Model Picker', 'Query by Committee', 'Structural Query by Committee', 'Random Sampling',
    #              'Importance Weighted Active Learning', 'Efficient Active Learning',
    #              "CAMS single policy","CAMS identity","CAMS test","contextual qbc","contextual iwal"])
    parser.add_argument("-which_methods",default=list([1,1,0,1,1,1,1,1,0,1,1]),nargs='+',type=int)
    parser.add_argument("-r",default=False,type=bool)

    # VERTEBRAL test setting
    parser.add_argument('-dataset', default="VERTEBRAL_contextual", type=str, help='dataset')
    parser.add_argument('-stream_size', default=130, type=int, help='stream size')
    parser.add_argument("-budget",default=[5,10, 50],nargs='+',help="budget",type=int)
    parser.add_argument("-num_reals", default=10, type=int, help="num reals")
    parser.add_argument("-p", default=[6], nargs='+', help="budget", type=int)  # best policy index, CAMS single policy only


    args = parser.parse_args()

    print(args)

    dataset = args.dataset
    stream_size = args.stream_size
    stream_setting= args.stream_setting
    budget=args.budget
    num_reals=args.num_reals
    which_methods=args.which_methods
    policy_index=args.p
    random_policy=args.r

    #query strategies
    if config.task == "task2":
        which_methods=[0,0,0,0,0,0,1,1,1,0,0]

    #obsolete task, for testing only
    if config.task == "task8":
        which_methods=[0,0,0,0,0,0,1,1,1,1,1]

    run_experiment(
        dataset,  # dataset,
        stream_size,  # pool_size,
        stream_setting,  # pool_setting,
        budget,  # query cost,
        num_reals,  # num_reals,
        which_methods,  # which_methods
        policy_index, #policy index
        random_policy #random policy
    )

