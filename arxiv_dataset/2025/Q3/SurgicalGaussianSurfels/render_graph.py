import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scene import Scene, GaussianModel, DeformModel
from gaussian_renderer import render
from utils.general_utils import safe_state
from utils.camera_utils import cameraList_from_camInfos
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args

# Helper to detect MLP type from config or directory name
MLP_TYPE_MAP = {
    'fullyfused': ("FullyFusedMLP", {"use_cutlass": False, "use_fullyfused": True}),
    'cutlass': ("CutlassMLP", {"use_cutlass": True, "use_fullyfused": False}),
    'pulling': ("RegularMLP", {"use_cutlass": False, "use_fullyfused": False}),
    'regular': ("RegularMLP", {"use_cutlass": False, "use_fullyfused": False}),
}

def detect_mlp_type(model_path):
    # Try to infer from config file first
    cfg_path = os.path.join(model_path, "cfg_args")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r") as f:
                cfg_text = f.read()
            # Look for fullyfusedMLP or cutlassMLP in the config text
            if "fullyfusedMLP=True" in cfg_text or "use_fullyfused=True" in cfg_text or "'use_fullyfused': True" in cfg_text:
                return MLP_TYPE_MAP['fullyfused']
            if "cutlassMLP=True" in cfg_text or "use_cutlass=True" in cfg_text or "'use_cutlass': True" in cfg_text:
                return MLP_TYPE_MAP['cutlass']
        except Exception as e:
            print(f"Warning: Could not parse cfg_args in {model_path}: {e}")
    # Fallback: Try to infer from directory name
    lower = os.path.basename(model_path).lower()
    for key, val in MLP_TYPE_MAP.items():
        if key in lower:
            return val
    # Fallback: default to RegularMLP
    return MLP_TYPE_MAP['regular']

def render_sets_with_timing(dataset, iteration, pipeline, skip_train, skip_test, mode, args, mlp_args):
    """Modified version of render_sets from render.py with timing measurements"""
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        # Create deformation model based on MLP flags
        deform = DeformModel(**mlp_args)
        deform.load_weights(dataset.model_path)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # Get test cameras for benchmarking
        views = scene.getTestCameras()
        
        # Time each frame
        times = []
        for idx, view in enumerate(tqdm(views, desc="Rendering with timing")):
            if dataset.load2gpu_on_the_fly:
                view.load2device()

            # Start GPU timer
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

            fid = view.fid
            xyz = gaussians.get_xyz
            time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
            d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)

            _ = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling)

            # Stop GPU timer
            end_event.record()
            torch.cuda.synchronize()
            frame_time = start_event.elapsed_time(end_event) / 1000  # seconds
            times.append(frame_time)
        
        return times

def main():
    # Get the models list first
    parser = ArgumentParser(description="Rendering speed benchmarking script for multiple models")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("models", nargs='+', help="List of model paths to benchmark")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--mode", default='render', choices=['render'])
    parser.add_argument("--mlp_types", nargs='*', default=None, help="List of MLP types for each model (fullyfused, cutlass, regular)")
    
    temp_args = parser.parse_args()
    models = temp_args.models
    print("Benchmarking rendering for models:", models)

    all_times = {}
    for i, model_path in enumerate(models):
        if temp_args.mlp_types and i < len(temp_args.mlp_types):
            mlp_type_key = temp_args.mlp_types[i].lower()
            mlp_name, mlp_args = MLP_TYPE_MAP.get(mlp_type_key, MLP_TYPE_MAP['regular'])
            print(f"MLP type manually set to: {mlp_name}")
        else:
            mlp_name, mlp_args = detect_mlp_type(model_path)
            print(f"Detected MLP type: {mlp_name}")
        
        # Create a separate parser for this model (avoiding conflicts)
        model_parser = ArgumentParser(description="Model-specific parser")
        model_params = ModelParams(model_parser, sentinel=True)
        pipeline_params = PipelineParams(model_parser)
        model_parser.add_argument("--iteration", default=-1, type=int)
        model_parser.add_argument("--mode", default='render', choices=['render'])
        
        # Create args with just the model path
        import sys
        original_argv = sys.argv.copy()
        sys.argv = [original_argv[0], "-m", model_path]
        args = get_combined_args(model_parser)
        
        # Extract parameters properly (following render.py pattern)
        dataset = model_params.extract(args)
        pipeline_config = pipeline_params.extract(args)
        
        try:
            times = render_sets_with_timing(dataset, args.iteration, pipeline_config, True, False, args.mode, args, mlp_args)
            all_times[os.path.basename(model_path) + f" ({mlp_name})"] = times
        except Exception as e:
            print(f"Skipping {model_path} due to error: {e}")
            continue
        
        # Restore original argv
        sys.argv = original_argv

    # Plotting
    if all_times:
        # Print performance data to terminal
        print("\n" + "="*60)
        print("RENDERING SPEED BENCHMARK RESULTS")
        print("="*60)
        
        # Get all unique iteration numbers
        max_iterations = max(len(times) for times in all_times.values())
        
        # Print header
        mlp_names = []
        for label in all_times.keys():
            if "FullyFusedMLP" in label:
                mlp_names.append("FFD-MLP")
            elif "CutlassMLP" in label:
                mlp_names.append("CutlassMLP")
            elif "RegularMLP" in label:
                mlp_names.append("RegularMLP")
            else:
                mlp_names.append(label)
        
        header = f"{'Iteration':<10}"
        for name in mlp_names:
            header += f"{name:>12}"
        print(header)
        print("-" * (10 + 12 * len(mlp_names)))
        
        # Print data for each iteration
        for i in range(max_iterations):
            row = f"{i+1:<10}"
            for label, times in all_times.items():
                if i < len(times):
                    fps = 1/times[i]
                    row += f"{fps:>12.1f}"
                else:
                    row += f"{'N/A':>12}"
            print(row)
        
        print("="*60)
        
        plt.figure(figsize=(7,5))
        for label, times in all_times.items():
            plt.plot(np.arange(1, len(times)+1), 1/np.array(times), label=label)  # FPS = 1/time
        plt.xscale('log')
        
        # Fix x-axis to show iteration numbers directly
        from matplotlib.ticker import ScalarFormatter
        plt.gca().xaxis.set_major_formatter(ScalarFormatter())
        plt.gca().xaxis.set_minor_formatter(ScalarFormatter())
        
        plt.xlabel('Iteration (frame index)')
        plt.ylabel('Speed (Frames per second)')
        #plt.title('Rendering Speed vs Iteration for Different Models/MLPs')
        
        # Fix legend to show only MLP types
        handles, labels = plt.gca().get_legend_handles_labels()
        new_labels = []
        for label in labels:
            if "FullyFusedMLP" in label:
                new_labels.append("FFD-MLP")
            elif "CutlassMLP" in label:
                new_labels.append("CutlassMLP")
            elif "RegularMLP" in label:
                new_labels.append("RegularMLP")
            else:
                new_labels.append(label)
        plt.legend(handles, new_labels)
        
        plt.grid(True, which="both", ls="--", lw=0.5)
        plt.tight_layout()
        plt.savefig('render_speed_comparison.png')
        plt.savefig('render_speed_comparison.pdf')  # Also save as PDF
        print("Saved graph to render_speed_comparison.png and render_speed_comparison.pdf")
    else:
        print("No models were successfully processed. No graph generated.")

if __name__ == "__main__":
    main() 