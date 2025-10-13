
import torch
import numpy as np
import argparse

from utils.utils import set_seed
from model.bm_imagenet import build_model_imagenet
from data.dataset_imagenet import build_dataloaders
from predictor.predictor import Predictor


def search_raps_para(conformal, loader, model, alpha, device):

    best_para = None
    min_size = float('inf') 

    start, end = 0.01, 0.20
    step = 0.01
    print(f"Searching in range ({start}, {end}) with step size {step}...")

    first_stage_paralist = [round(x * step, 4) for x in range(int(start / step), int(end / step) + 1)]
    best_first_stage_para = None
    for para in first_stage_paralist:
        predictor = Predictor(model, conformal, alpha, device)
        predictor.score_function.penalty = para
        
        predictor.calibrate(loader)
        
        acc, cov, size = predictor.evaluate(loader)
        
        print(f"Parameter: {para}, Size: {size}")
        
        if size < min_size:
            min_size = size
            best_para = para
            best_first_stage_para = para
    

    if best_first_stage_para is not None:
        print(f"Best parameter from first stage: {best_first_stage_para}")
        
        second_stage_start = best_first_stage_para - 0.005
        second_stage_end = best_first_stage_para + 0.005
        second_stage_step = 0.001 
        
        print(f"Searching in range ({second_stage_start}, {second_stage_end}) with step size {second_stage_step}...")
        second_stage_paralist = [round(x * second_stage_step, 4) for x in range(int(second_stage_start / second_stage_step), int(second_stage_end / second_stage_step) + 1)]

        for para in second_stage_paralist:
            predictor = Predictor(model, conformal, alpha, device)
            predictor.score_function.penalty = para
            
            predictor.calibrate(loader)
            

            acc, cov, size = predictor.evaluate(loader)
            
            print(f"Parameter: {para}, Size: {size}")
            
            if size < min_size:
                min_size = size
                best_para = para
    
    print(f"Best parameter: {best_para}")
    return best_para



def run_experiment(model_name, cal_num, conformal, alpha, num_runs, device):
    # Initialize lists to store results for accuracy, coverage, and size
    all_acc_ori = []
    all_cov_ori = []
    all_size_ori = []

    all_acc_dz = []
    all_cov_dz = []
    all_size_dz = []


    for run in range(num_runs):
        print(f"Running experiment {run + 1}/{num_runs}...")
        
        set_seed(run+66)  
        
        model = build_model_imagenet(model_name)
        model = model.to(device)
        
        # Assuming the dataset is the same for both train and calibrate
        tuning_calibloader, cp_calibloader, testloader = build_dataloaders('/data/dataset/', cal_num)
        

     
        optimal_para = search_raps_para(conformal, cp_calibloader, model, alpha, device)
        predictor = Predictor(model, conformal, alpha, device)
        predictor.score_function.penalty = optimal_para

        optimal_para_dz = search_raps_para(conformal, tuning_calibloader, model, alpha, device)
        predictor_dz = Predictor(model, conformal, alpha, device)
        predictor_dz.score_function.penalty = optimal_para_dz 

    
        # Calibration using cp_calibloader
        predictor.calibrate(cp_calibloader)
        acc, cov, size = predictor.evaluate(testloader)

        predictor_dz.calibrate(cp_calibloader)
        acc_dz, cov_dz, size_dz = predictor_dz.evaluate(testloader)

        print(f"Result for run {run + 1}:")
        print(f"Same train and calibrate set - Accuracy: {acc}, Coverage_rate: {cov}, Size: {size}")
        print(f"Different train and calibrate set - Accuracy: {acc_dz}, Coverage_rate: {cov_dz}, Size: {size_dz}")

        # Collect results for each experiment
        all_acc_ori.append(acc)
        all_cov_ori.append(abs(cov - (1-alpha)))
        all_size_ori.append(size)

        all_acc_dz.append(acc_dz)
        all_cov_dz.append(abs(cov_dz - (1-alpha)))
        all_size_dz.append(size_dz)
    
    return (all_acc_ori, all_cov_ori, all_size_ori), (all_acc_dz, all_cov_dz, all_size_dz)



def save_experiment_log(config, results_ori, results_extra, filename):

    with open(filename, 'a') as f:  # 使用 'a' 模式，追加内容
        for key, value in config.items():
            f.write(f'{key} = {value}\n')
        
        acc_ori, cov_ori, size_ori = map(np.array, results_ori)
        acc_extra, cov_extra, size_extra = map(np.array, results_extra)

        mean_acc_ori = np.mean(acc_ori)
        std_acc_ori = np.std(acc_ori)

        mean_cov_ori = np.mean(cov_ori)
        std_cov_ori = np.std(cov_ori)

        mean_size_ori = np.mean(size_ori)
        std_size_ori = np.std(size_ori)

        mean_acc_extra = np.mean(acc_extra)
        std_acc_extra = np.std(acc_extra)

        mean_cov_extra = np.mean(cov_extra)
        std_cov_extra = np.std(cov_extra)

        mean_size_extra = np.mean(size_extra)
        std_size_extra = np.std(size_extra)

        f.write("\nSummary of Results:\n")
        f.write(f"Predictor_ori - Mean Accuracy: {mean_acc_ori}, Std: {std_acc_ori}\n")
        f.write(f"Predictor_ori - Mean Coverage_Gap: {mean_cov_ori}, Std: {std_cov_ori}\n")
        f.write(f"Predictor_ori - Mean Size: {mean_size_ori}, Std: {std_size_ori}\n")
        
        f.write(f"Predictor_extra - Mean Accuracy: {mean_acc_extra}, Std: {std_acc_extra}\n")
        f.write(f"Predictor_extra - Mean Coverage_Gap: {mean_cov_extra}, Std: {std_cov_extra}\n")
        f.write(f"Predictor_extra - Mean Size: {mean_size_extra}, Std: {std_size_extra}\n")
        
        f.write("="*50 + "\n\n")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run an experiment with configurable parameters.')
    
    # Command-line arguments
    parser.add_argument('--cal_num', type=int, default=5000, help='Number of calibration samples')
    parser.add_argument('--num_classes', type=int, default=1000, help='Number of classes')
    parser.add_argument('--conformal', type=str, default='raps', help='Conformal method')
    parser.add_argument('--alpha', type=float, default=0.1, help='Significance level')
    parser.add_argument('--num_runs', type=int, default=20, help='Number of runs')
    parser.add_argument('--file', type=str, help='Result file name')
    parser.add_argument('--preprocess', type=str, default='ts', help='Preprocessing method')
    parser.add_argument('--device', type=str, default='cuda:7', help='GPU device to use (e.g., cuda:0, cuda:1, or cpu)')

    args = parser.parse_args()

    # Set device for model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Model and configuration setup
    model_name = 'resnet18'
    
    # Construct configuration dictionary
    config = {
        'cal_num': args.cal_num,
        'num_classes': args.num_classes,
        'conformal': args.conformal,
        'alpha': args.alpha,
        'preprocess': args.preprocess
    }
    # Run multiple experiments and collect results
    (all_acc_ori, all_cov_ori, all_size_ori), (all_acc_dz, all_cov_dz, all_size_dz) = run_experiment(
        model_name, 
        args.cal_num,     
        args.conformal,   
        args.alpha,       
        args.num_runs,    
        device            
    )

    # Summarize and save the results
    save_experiment_log(config, 
                        (all_acc_ori, all_cov_ori, all_size_ori), 
                        (all_acc_dz, all_cov_dz, all_size_dz),
                        args.file)

