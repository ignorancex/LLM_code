import os
import torch
import time
import json

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess PDF files for VQA generation.")
    parser.add_argument('--workspace', type=str, required=True, help='Path to workspace folder')
    parser.add_argument('--cuda_visible_devices', type=str, default='0', help='CUDA visible devices')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--HealthGPT_ViT_path', type=str, default='', help='Path to the VIT model for HealthGPT')
    parser.add_argument('--model_format', type=str, default='awq', help='Model format')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for DataLoader')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for DataLoader')
    parser.add_argument('--reasoning', action='store_true', help='Whether to generate reasoning response')
    parser.add_argument('--parse_reasoning', action='store_true', help='Whether to ONLY parse reasoning response and skip the inference')
    return parser.parse_args()
    
def main():
    args = parse_args()
    reasoning = args.reasoning
    print(args.parse_reasoning)
    print(args.reasoning)

    workspace = args.workspace
    os.makedirs(os.path.join(workspace, "results"), exist_ok=True)
    with open(os.path.join(workspace, "data.json"), "r") as f:
        all_MC = json.load(f)

    if not reasoning:
        res_path = os.path.join(workspace, "results", f"MCresults_{args.model_path.split('/')[-1]}.jsonl")
    else:
        res_path = os.path.join(workspace, "results", f"reasoning_MCresults_{args.model_path.split('/')[-1]}.jsonl")
    if not os.path.exists(res_path):
        with open(res_path, 'w') as f:
            pass
    with open(res_path, 'r') as f:
        finished = [json.loads(line) for line in f.readlines()]
        finished = [item["_id"] for item in finished]
    all_MC = [item for item in all_MC if item["_id"] not in finished]

    print(f"Number of questions to evaluate: {len(all_MC)}")
    if len(all_MC) == 0:
        return

    # import functions
    from eval_utils import evaluate_MC, parse_reasoning_response

    # perform evaluation inference
    if not args.parse_reasoning:
        if not "API" in args.model_path:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Device: {device}")

        if "API" in args.model_path:
            from models.openAI import OpenAI_API
            model = OpenAI_API(args)
            device = torch.device("cpu")

        elif "InternVL" in args.model_path:
            from models.InternVL import InternVL
            model = InternVL(args)

        elif "llava-onevision-qwen2" in args.model_path:
            from models.LLaVA_Qwen import LLaVA_Qwen
            model = LLaVA_Qwen(args)

        elif "Ovis2" in args.model_path:
            from models.Ovis import Ovis
            model = Ovis(args)

        elif "Qwen2.5-VL" in args.model_path or "VL-Reasoner" in args.model_path:
            from models.QwenVL import QwenVL_model
            model = QwenVL_model(args)

        elif "InternVL2_5" in args.model_path:
            from models.lmdeploy import lmdeploy_model
            model = lmdeploy_model(args)

        elif "deepseek-vl2" in args.model_path:
                from models.DeepSeekVL2 import DeepSeekVL2
                model = DeepSeekVL2(args)

        elif "Skywork-R1V" in args.model_path:
            from models.Skywork_R1V import Skywork_R1V
            model = Skywork_R1V(args)

        elif "Kimi-VL" in args.model_path:
            from models.Kimi_VL import KimiVL
            model = KimiVL(args)

        elif "HuatuoGPT" in args.model_path:
            from models.HuatuoGPT import HuatuoGPT
            model = HuatuoGPT(args)

        elif "HealthGPT" in args.model_path:
            print("Go to HealthGPTmodel/llava/DOABbench_test")

        elif "MedVLM-R1" in args.model_path:
            from models.MedVLM_R1 import MedVLM_R1
            model = MedVLM_R1(args)
            
        else:
            raise ValueError(f"Model {args.model_path} not supported.")
            
        evaluate_MC(all_MC, res_path, model, args)

    
    parsed_path = os.path.join(workspace, "results", f"MCresults_{args.model_path.split('/')[-1]}.jsonl")
    error_path = os.path.join(workspace, "results", f"reasoning_error_{args.model_path.split('/')[-1]}.json")
    if args.reasoning or args.parse_reasoning:
        parse_reasoning_response(res_path, parsed_path, error_path)
    return


if __name__ == "__main__":
    main()