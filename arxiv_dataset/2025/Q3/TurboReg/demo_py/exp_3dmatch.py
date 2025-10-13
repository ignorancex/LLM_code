import time
from typing import Literal
import tyro
import torch
from dataclasses import dataclass

from .dataset_3dmatch import TDMatchFCGFAndFPFHDataset
from demo_py.utils_pcr import compute_transformation_error, numpy_to_torch32
import turboreg_gpu 


@dataclass
class Args:
    # Dataset path
    dir_dataset: str
    desc: Literal["fpfh", "fcgf"] = "fcgf"
    dataname: Literal["3DMatch", "3DLoMatch"] = "3DLoMatch"

    # TurboRegGPU Initialization Parameters
    max_N: int = 7000
    tau_length_consis: float = 0.012
    num_pivot: int = 2000
    radiu_nms: float = 0.15
    tau_inlier: float = 0.1
    metric_str: Literal["IN", "MSE", "MAE"] = "IN" 

def main():
    args = tyro.cli(Args)

    if args.dataname.lower() == "3dmatch":
        processed_dataname = "3DMatch"
    elif args.dataname.lower() == "3dlomatch":
        processed_dataname = "3DLoMatch"
    else:
        raise ValueError(f"Invalid dataname: {args.dataname}. Expected '3DMatch' or '3DLoMatch'.")

    # TurboReg
    reger = turboreg_gpu.TurboRegGPU(
        args.max_N,
        args.tau_length_consis,
        args.num_pivot,
        args.radiu_nms,
        args.tau_inlier,
        args.metric_str
    )

    ds = TDMatchFCGFAndFPFHDataset(base_dir=args.dir_dataset, dataset_type=processed_dataname, descriptor_type=args.desc)

    num_succ = 0
    for i in range(len(ds)):
        data = ds[i]
        kpts_src, kpts_dst, trans_gt = data['kpts_src'], data['kpts_dst'], data['trans_gt']
        
        # Move keypoints to CUDA device
        kpts_src, kpts_dst = numpy_to_torch32(
            torch.device('cuda:0'), kpts_src, kpts_dst
        )

        # Run TurboReg
        t1 = time.time()
        trans_pred_torch = reger.run_reg(kpts_src, kpts_dst)
        T_reg = (time.time() - t1) * 1000
        trans_pred = trans_pred_torch.cpu().numpy()
        rre, rte = compute_transformation_error(trans_gt, trans_pred)
        is_succ = (rre < 15) & (rte < 0.3)
        num_succ += is_succ
        
        print(f"Processed item {i+1}/{len(ds)}: Registration time: {T_reg:.3f} ms, RR= {(num_succ / (i+1)) * 100:.3f}%")

if __name__ == "__main__":
    main()

"""
python -m demo_py.exp_3dmatch --desc fpfh --dataname 3DMatch --dir_dataset "DIR_3DMATCH_FPFH_FCGF" --max_N 7000 --tau_length_consis 0.012 --num_pivot 2000 --radiu_nms 0.15 --tau_inlier 0.1 --metric_str "IN"
python -m demo_py.exp_3dmatch --desc fcgf --dataname 3DMatch --dir_dataset "DIR_3DMATCH_FPFH_FCGF" --max_N 6000 --tau_length_consis 0.012 --num_pivot 2000 --radiu_nms 0.10 --tau_inlier 0.1 --metric_str "MAE"
"""