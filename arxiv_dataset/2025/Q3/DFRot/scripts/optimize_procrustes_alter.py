import os
import pickle
import sys
from functools import partial

import numpy as np
import torch
import tqdm

sys.path.append("..")
from utils.quant_utils import ActQuantizer
from utils.hadamard_utils import random_hadamard_matrix
import argparse


def to_numpy(data):
    return data.detach().cpu().numpy()


def print_log(print_string, log):
    print("{:}".format(print_string))
    log.write('{:}\n'.format(print_string))
    log.flush()


def quant_func(x, n_bits, sym=True, clip_ratio=1.0):
    quantizer = ActQuantizer()
    quantizer.configure(n_bits, groupsize=-1, sym=sym, clip_ratio=clip_ratio)
    quantizer.find_params(x)
    x = quantizer(x)
    quantizer.free()
    return x


def is_pow2(n):
    return (n & (n - 1) == 0) and (n > 0)


def orthogonal_procrustes(A, B):
    C = torch.matmul(A.T, B)
    U, _, Vt = torch.linalg.svd(C, full_matrices=True)
    R = torch.matmul(U, Vt)

    return R


@torch.no_grad()
def get_best_rotate_via_procrustes(A_init, log_func, n_bits=4, steps=1000, show_init=False, clip_ratio=1.0, args=None):
    # assert is_pow2(A_init.shape[-1]), f"A.shape[-1]={A_init.shape[-1]} is not pow of two is not support!"
    shape = A_init.shape[-1]
    if show_init:
        # Initialize random hadamard matrix and randomized orthogonal matrix via QR decomposition
        hadamard = random_hadamard_matrix(shape, A_init.device).to(dtype=torch.float32)
        R = torch.randn(shape, shape)
        R_random = torch.linalg.qr(R)[0].to(A_init)
        # get init quantization error
        init_loss = torch.norm(A_init - quant_func(A_init, n_bits, clip_ratio=clip_ratio))
        log_func(f"Init Quantization Error: {init_loss:.5f}")
        # X @ hadamard
        hadamard_x1 = A_init @ hadamard - quant_func(A_init @ hadamard, n_bits, clip_ratio=clip_ratio)
        hadamard_x1_loss = torch.norm(hadamard_x1)
        log_func(f"Hadamard Quantization Error: {hadamard_x1_loss:.5f}")
        #
        random_loss = torch.norm(A_init @ R_random - quant_func(A_init @ R_random, n_bits, clip_ratio=clip_ratio))
        log_func(f"Random Quantization Error: {random_loss:.5f}")
        if args.rotate_mode == 'hadamard':
            return hadamard
        elif args.rotate_mode == 'random':
            return R_random
        else:
            raise NotImplementedError

    # Random hadamard matrix is a good beginning
    A = A_init
    R_accumulate = torch.eye(shape).to(A_init)
    best_loss = torch.norm(A - quant_func(A, n_bits=n_bits, clip_ratio=clip_ratio)) + 100

    # Procrustes Optimization
    for step in tqdm.tqdm(range(1, steps + 1), desc="(Procrustes) Optimization"):
        A_ = A @ R_accumulate
        B_ = quant_func(A_, n_bits=n_bits, clip_ratio=clip_ratio)
        R = torch.linalg.qr(orthogonal_procrustes(A_, B_))[0]
        loss = torch.norm(A @ R_accumulate @ R - quant_func(A @ R_accumulate @ R, n_bits=n_bits, clip_ratio=clip_ratio))
        log_func(f"Step: {step} Loss: {loss:.5f}")
        if loss <= best_loss and (best_loss - loss) > 0.1:
            best_loss = loss
            R_accumulate = R_accumulate @ R
        else:
            break

    return R_accumulate


def get_data(saved_path):
    # read saved
    with open(saved_path, 'rb') as f:
        loaded_dict = pickle.load(f)
    # stack data
    output = np.concatenate([value[0].squeeze() for _, value in loaded_dict.items()], axis=0)
    return output


if __name__ == "__main__":
    if not torch.cuda.is_available():
        template = ("python3 optimize_procrustes_alter.py --rotate_mode {rotate_mode} "
                    "--data_principle {data_principle} --alpha {alpha}")
        rotate_modes = ['random', 'hadamard']
        data_principles = ['alter', ]
        alphas = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
        for rotate_mode in rotate_modes:
            for data_principle in data_principles:
                for alpha in alphas:
                    print(template.format(rotate_mode=rotate_mode, data_principle=data_principle, alpha=alpha))
        exit()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    parser = argparse.ArgumentParser()

    # General Arguments
    parser.add_argument('--rotate_mode', type=str, default='hadamard', choices=['hadamard', 'random'])
    parser.add_argument('--data_principle', type=str, default='alter', choices=['alter', ])
    parser.add_argument('--alpha', type=float, default=100, help='massive activation importance')

    args = parser.parse_args()

    device = 'cuda:0'
    # get your calibration_dir
    calibration_dir = "../DFRot/calibration"
    model_names = ['LLaMA-2-7B', 'LLaMA-3-8B', 'LLaMA-2-13B', "Mistral-7B", "Mistral-7B-V3", 'QWen2-7B']
    model_names = [model_names[-1], ]
    calibrate_files = ["Llama-2-7b-hf.pkl", "Meta-Llama-3-8B.pkl", "Llama-2-13b-hf.pkl", "Mistral-7B-v0.1.pkl",
                       "Mistral-7B-v0.3.pkl", 'Qwen2-7B.pkl']
    calibrate_files = [calibrate_files[-1], ]
    alter_num = 100
    n_bits = 4
    clip_ratio = 1.0

    for model_name, calibrate_file in zip(model_names, calibrate_files):
        # mkdir log_dir
        data_dir = os.path.join(f"../rms_norm_feature_{args.rotate_mode}_{args.data_principle}", f"{args.alpha:.0f}")
        os.makedirs(data_dir, exist_ok=True)
        log = open(os.path.join(data_dir, '{}_procrustes.txt'.format(model_name)), 'w')
        log_func = partial(print_log, log=log)
        log_func(f"{model_name}")

        # load calibration datasets
        calibrate_data_path = os.path.join(calibration_dir, calibrate_file)
        data_collects = get_data(calibrate_data_path)
        shape = data_collects.shape[-1]

        A_init = torch.from_numpy(data_collects).to(device=device, dtype=torch.float32)
        StartR = get_best_rotate_via_procrustes(A_init=A_init, n_bits=n_bits, log_func=log_func, steps=0,
                                                show_init=True, clip_ratio=clip_ratio, args=args)

        # show hadamard baseline
        _A = A_init @ StartR
        error_A_init = torch.norm(A_init - quant_func(A_init, n_bits=n_bits), dim=-1)
        R_random = torch.linalg.qr(torch.randn(shape, shape))[0].to(A_init)
        error_random = torch.norm(A_init @ R_random - quant_func(A_init @ R_random, n_bits=n_bits), dim=-1)
        R_optimize = torch.eye(shape).to(device)
        assert args.data_principle == 'alter' and alter_num >= 1

        for idx in range(alter_num):
            # Different LLM has different massive activation quantization error threshold
            if 'Mistral-7B' not in model_name:
                threshold = 8.5
                A = torch.where(error_A_init.reshape(-1, 1) < threshold, _A  * args.alpha, _A)
                num = torch.sum(error_A_init.reshape(-1, 1) < threshold)
            else:
                A = torch.where(error_random.reshape(-1, 1) < error_A_init.reshape(-1, 1), _A, _A * args.alpha)
                num = torch.sum(error_random.reshape(-1, 1) < error_A_init.reshape(-1, 1))
            optimize_max_num = 1
            log_func(
                f"{idx}-step the data shape is: {A.shape} {num}")
            log_func(
                f"Initialize: {torch.norm(A @ R_optimize - quant_func(A @ R_optimize, n_bits, True, clip_ratio)):.5f}")
            procrustes = get_best_rotate_via_procrustes(
                A @ R_optimize, log_func, n_bits=n_bits,
                steps=optimize_max_num, clip_ratio=clip_ratio, args=args
            )
            R_optimize = R_optimize @ procrustes
            R_optimize = torch.linalg.qr(R_optimize)[0]

        # Get optimized R
        R_optimize = StartR @ R_optimize
        log_func(f"Sum of Diag(R @ R.T):  {torch.sum(torch.diag(R_optimize @ R_optimize.T))}")

        loss = torch.norm(A_init @ R_optimize - quant_func(A_init @ R_optimize, n_bits=n_bits, clip_ratio=clip_ratio))
        log_func(f"Final Quantization Error: {loss:.5f}")

        # save R to numpy
        R_optimize = to_numpy(R_optimize)
        save_path = os.path.join(data_dir, model_name + f"-{n_bits}.npy")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, R_optimize)
        log.close()
