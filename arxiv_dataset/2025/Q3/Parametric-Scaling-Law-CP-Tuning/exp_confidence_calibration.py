import argparse
import torch
import numpy as np

from utils.utils import set_seed
from model.bm_imagenet import build_model_imagenet
from data.dataset_imagenet import build_dataloaders
from predictor.predictor import Predictor
from preprocessor.preprocesser import build_preprocessor

def run_experiment(model_name, preprocess, cal_num, num_classes, conformal, alpha, num_runs, freeze_num, device):
    # Initialize lists to store results for accuracy, coverage, and size
    all_acc_ori = []
    all_cov_ori = []
    all_size_ori = []

    all_acc_dz = []
    all_cov_dz = []
    all_size_dz = []

    for run in range(num_runs):
        print(f"Running experiment {run + 1}/{num_runs}...")
        
        set_seed(run+666)  
        
        model = build_model_imagenet(model_name)
        model = model.to(device)
        
        # Assuming the dataset is the same for both train and calibrate
        cc_calibloader, cp_calibloader, testloader = build_dataloaders('/data/dataset/', cal_num)
        
        # Preprocessing and predictor with the same train and calibrate set
        preprocessor = build_preprocessor(preprocess, model, device, num_classes,freeze_num)
        preprocessor.train(cp_calibloader)  # Train on cp_calibloader
        predictor = Predictor(model, conformal, alpha, device , preprocessor)

        # Preprocessing and predictor with different train and calibrate sets
        preprocessor_dz = build_preprocessor(preprocess, model, device, num_classes,freeze_num)
        preprocessor_dz.train(cc_calibloader)  # Train on cc_calibloader
        predictor_dz = Predictor(model, conformal, alpha, device, preprocessor_dz)
    
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

    with open(filename, 'a') as f:
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
    parser.add_argument('--conformal', type=str, default='thr', help='Conformal method')
    parser.add_argument('--alpha', type=float, default=0.1, help='Significance level')
    parser.add_argument('--num_runs', type=int, default=30, help='Number of runs')
    parser.add_argument('--freeze_num', type=int, default=0, help='free_num')
    parser.add_argument('--preprocess', type=str, default='vs', help='Preprocessing method')
    parser.add_argument('--file', type=str, help='Result file name')
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
        'preprocess': args.preprocess,
        'freeze_num': args.freeze_num
    }

    # Run multiple experiments and collect results
    (all_acc_ori, all_cov_ori, all_size_ori), (all_acc_dz, all_cov_dz, all_size_dz) = run_experiment(
        model_name, 
        args.preprocess, 
        args.cal_num,     
        args.num_classes, 
        args.conformal,   
        args.alpha,       
        args.num_runs,  
        args.freeze_num,  
        device            
    )

    # Summarize and save the results
    save_experiment_log(config, 
                        (all_acc_ori, all_cov_ori, all_size_ori), 
                        (all_acc_dz, all_cov_dz, all_size_dz),
                        args.file)

