import os
import numpy as np

from CrowdES.inference_model import CrowdESFramework
from utils.dataloader.evaluation_dataloader import EvaluationDataset
from utils.utils import reproducibility_settings
from utils.trajectory import image_to_world_trajectory, export_generated_scenario_to_socialgan_data
from utils.visualization import produce_video_from_data


# Global settings
TRIALS = 1                      # Number of trials for each scene
SCENARIO_LENGTH = 30 * 60 * 10  # Scenario length in frames (30fps * 60s * 10min), if None, use the length of the scene
POSTFIX = 'crowdes'             # Postfix for the generated scenario files
EXPORT_SPATIAL_LAYOUT = True    # Export predicted spatial layout
EXPORT_SOCIALGAN_DATA = True    # Export to text file for trajectory prediction model training
EXPORT_VIDEO = True             # Export video for visualization


def main(config, seed=0):
    # Reproducibility
    reproducibility_settings(seed=seed)

    # Load dataset and framework
    dataset_test = EvaluationDataset(config, 'test')
    framework = CrowdESFramework(config)

    # Start inference
    dataset_name = config.dataset.dataset_name
    scene_list = dataset_test.scene_list

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
            scenario_length = data['size']['length'] if SCENARIO_LENGTH is None else SCENARIO_LENGTH
            framework.initialize_scene(scene_img, scene_seg, scene_walkable, scene_navmesh, scene_H)
            generated_scenario = framework.generate(scenario_length, seed=trial)
            generated_scenario['scene'] = scene
            print(f'Generated scenario: {len(generated_scenario['agent_id'].unique())} total agents')

            # Save generated scenario
            os.makedirs(f'./output/generated/{dataset_name}', exist_ok=True)
            columns=['scene', 'agent_id', 'agent_type', 'frame', 'x', 'y']
            generated_scenario = generated_scenario[columns]
            generated_scenario.to_csv(f'./output/generated/{dataset_name}/{scene}-{POSTFIX}-{trial}.csv', index=False)
            # generated_scenario_meter = generated_scenario.copy()
            # generated_scenario_meter.loc[:, ['x', 'y']] = image_to_world_trajectory(generated_scenario_meter[['x', 'y']].values, scene_H)
            # generated_scenario_meter.to_csv(f'./output/generated/{dataset_name}/{scene}-{POSTFIX}-{trial}-meter.csv', index=False)

            # Export predicted spatial layout
            if EXPORT_SPATIAL_LAYOUT:
                from PIL import Image
                appearance_density_map_raw = framework.appearance_density_map_raw
                appearance_density_map_raw = (appearance_density_map_raw * 255).astype(np.uint8)
                appearance_density_map_raw = Image.fromarray(appearance_density_map_raw)
                appearance_density_map_raw.save(f'./output/generated/{dataset_name}/{scene}_appearance_density.png')

                population_density_map_raw = framework.population_density_map_raw
                population_density_map_raw = (population_density_map_raw * 255).astype(np.uint8)
                population_density_map_raw = Image.fromarray(population_density_map_raw)
                population_density_map_raw.save(f'./output/generated/{dataset_name}/{scene}_population_density.png')
                
                scene_bg = data['bg']
                scene_bg.save(f'./output/generated/{dataset_name}/{scene}_scene_img.png')

            # Export generated scenario to social-gan style dataset file for trajectory prediction model training
            if EXPORT_SOCIALGAN_DATA:
                export_path = f'./output/generated/{dataset_name}/{scene}-{POSTFIX}-{trial}.txt'
                os.makedirs(os.path.dirname(export_path), exist_ok=True)
                export_generated_scenario_to_socialgan_data(export_path, generated_scenario, scene_H)
            print(f'Social-gan style scenario saved at {export_path}')

            # Export video for visualization
            if EXPORT_VIDEO:
                scene_bg = data['bg']
                video_path = f'./output/generated/{dataset_name}/{scene}-{POSTFIX}-{trial}.avi'
                os.makedirs(os.path.dirname(video_path), exist_ok=True)
                produce_video_from_data(video_path, scene_img, scene_bg, generated_scenario, scenario_length, config)
                print(f'Video saved at {video_path}')

    return 
