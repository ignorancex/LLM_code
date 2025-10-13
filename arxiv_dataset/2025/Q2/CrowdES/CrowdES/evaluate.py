import os
import json
import numpy as np

from CrowdES.inference_model import CrowdESFramework
from utils.dataloader.evaluation_dataloader import EvaluationDataset
from utils.utils import reproducibility_settings
from utils.metrics import compute_metrics
from utils.visualization import produce_video_from_data


# Global settings
TRIALS = 20           # Number of trials for each scene. DO NOT CHANGE THIS!
EXPORT_VIDEO = False  # Export video for visualization


def main(config, seed=0):
    # Reproducibility
    reproducibility_settings(seed=seed)

    # Load dataset and framework
    dataset_test = EvaluationDataset(config, 'test')
    framework = CrowdESFramework(config)

    # Start inference
    dataset_name = config.dataset.dataset_name
    scene_list = dataset_test.scene_list

    all_metrics = {}
    for scene_idx, scene in enumerate(scene_list):
        for trial in range(TRIALS):
            print(f'Scene {scene_idx + 1}/{len(scene_list)}: {scene}, Trial {trial + 1}/{TRIALS}')

            # Load data
            data = dataset_test[scene_idx]
            scene_img = data['img']
            scene_seg = data['seg']
            scene_H = data['H']
            scene_walkable = data['walkable']
            scene_navmesh = data['navmesh']

            # Inference
            scenario_length = data['size']['length']
            framework.initialize_scene(scene_img, scene_seg, scene_walkable, scene_navmesh, scene_H)
            generated_scenario = framework.generate(scenario_length, seed=trial)
            generated_scenario['scene'] = scene

            # Evaluation
            # Print statistics
            gt_agents = data['size']['num_agents']
            print(f'GT scenario: {gt_agents} total agents')
            print(f'Generated scenario: {len(generated_scenario['agent_id'].unique())} total agents')

            # Compute metrics
            scene_size = data['size']
            scene_trajectory_dense = data['trajectory_dense']
            metrics = compute_metrics(generated_scenario, scene_trajectory_dense, scene_size, scene_H)
            print(f'Metrics: {metrics}')

            for key, value in metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = {}
                for k, v in value.items():
                    if k not in all_metrics[key]:
                        all_metrics[key][k] = []
                    all_metrics[key][k].append(v)
            
            # Export video for visualization
            if EXPORT_VIDEO:
                scene_bg = data['bg']
                video_path = f'./output/generated/{dataset_name}/{scene}-{trial}.avi'
                os.makedirs(os.path.dirname(video_path), exist_ok=True)
                produce_video_from_data(video_path, scene_img, scene_bg, generated_scenario, scenario_length, config)
                print(f'Video saved at {video_path}')

    # Print average metrics
    average_metrics = {}
    print('Average metrics:')
    for key, value in all_metrics.items():
        average_metrics[key] = {}
        for k, v in value.items():
            print(f'{key} {k}: {np.mean(v)}')
            average_metrics[key][k] = np.mean(v)

    # Save metrics to file
    metrics_path = f'./output/log/{dataset_name}/metrics_{dataset_name}_{TRIALS}.json'
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(average_metrics, f, indent=4)
    print(f'Metrics saved at {metrics_path}')

    return average_metrics
