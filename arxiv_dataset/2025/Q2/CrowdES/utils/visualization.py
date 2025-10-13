import os
import cv2
import numpy as np
from tqdm.auto import tqdm


def draw_filled_circle(image, center, size, fill_color, edge_color, line_thickness):
    """Draw a filled circle with the center and size"""

    half_size = int(size // 2)
    cv2.circle(image, center, half_size, fill_color, -1)  # Fill the circle
    cv2.circle(image, center, half_size, edge_color, line_thickness)  # Draw the circle edge


def draw_filled_triangle(image, center, size, fill_color, edge_color, line_thickness):
    """Draw an equilateral triangle with all sides equal to size"""

    height = size * (np.sqrt(3) / 2)
    half_size = size / 2
    vertices = np.array([
        [center[0], center[1] - height / 3 * 2],  # Top vertex
        [center[0] - half_size, center[1] + height / 3],  # Bottom left
        [center[0] + half_size, center[1] + height / 3],  # Bottom right
    ], np.int32)
    vertices = vertices.reshape((-1, 1, 2))
    
    cv2.fillPoly(image, [vertices], fill_color)  # Fill the triangle
    cv2.polylines(image, [vertices], isClosed=True, color=edge_color, thickness=line_thickness)  # Draw the triangle edge


def draw_filled_square(image, center, size, fill_color, edge_color, line_thickness):
    """Define the vertices of the square with the center and size"""

    half_size = size // 2
    vertices = np.array([
        [center[0] - half_size, center[1] - half_size],  # Top left
        [center[0] + half_size, center[1] - half_size],  # Top right
        [center[0] + half_size, center[1] + half_size],  # Bottom right
        [center[0] - half_size, center[1] + half_size],  # Bottom left
    ], np.int32)
    vertices = vertices.reshape((-1, 1, 2))
    
    cv2.fillPoly(image, [vertices], fill_color)  # Fill the square
    cv2.polylines(image, [vertices], isClosed=True, color=edge_color, thickness=line_thickness)  # Draw the square edge


def produce_video_from_data(video_path, scene_img, scene_bg, generated_scenario, scenario_length, config):
    """
    Produce a video from the generated scenario data.
    
    Params:
        video_path (str): Path to save the video.
        scene_img (np.ndarray): The background image of the scene.
        scene_bg (np.ndarray): The background image of the scene.
        generated_scenario (pd.DataFrame): DataFrame containing the generated scenario data.
        scenario_length (int): Length of the scenario in frames.
        config (Config): Configuration object containing simulation parameters.
    """

    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    fill_colors = [(234, 56, 41), (246, 133, 17), (248, 217, 4), (170, 216, 22), (39, 187, 54), (0, 143, 93), (15, 181, 174), 
                   (51, 197, 232), (56, 146, 243), (104, 109, 244), (137, 61, 231), (224, 85, 226), (222, 61, 130)]
    edge_color = (177, 177, 177)
    
    line_thickness = max(1, round(max(scene_img.shape[0], scene_img.shape[1]) / 720 * 1))
    circle_size = max(10, round(max(scene_img.shape[0], scene_img.shape[1]) / 720 * 20))
    triangle_size = max(10, round(max(scene_img.shape[0], scene_img.shape[1]) / 720 * 20))
    square_size = max(8, round(max(scene_img.shape[0], scene_img.shape[1]) / 720 * 16))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = config.crowd_simulator.simulator.simulator_fps
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (scene_img.shape[1], scene_img.shape[0]))
    for frame in tqdm(range(scenario_length), desc='Generating video'):
        if frame % (config.dataset.dataset_fps // config.crowd_simulator.simulator.simulator_fps) != 0:
            # continue  # skip frames to reduce video size
            pass  # include all frames

        # Draw the background image
        frame_img = np.array(scene_bg).copy()
        frame_img = cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR)

        frame_agents = generated_scenario[generated_scenario['frame'] == frame]
        for agent_idx, agent in frame_agents.iterrows():
            agent_id = agent['agent_id']
            agent_type = agent['agent_type']
            agent_x, agent_y = int(agent['x']), int(agent['y'])
            agent_color = fill_colors[agent_id % len(fill_colors)][::-1]
            # draw agent on frame_img
            if agent_type == 0:  # pedestrian
                draw_filled_circle(frame_img, (agent_x, agent_y), circle_size, agent_color, edge_color, line_thickness)
            elif agent_type == 1:  # rider
                draw_filled_triangle(frame_img, (agent_x, agent_y), triangle_size, agent_color, edge_color, line_thickness)
            elif agent_type == 2:  # vehicle
                draw_filled_square(frame_img, (agent_x, agent_y), square_size, agent_color, edge_color, line_thickness)
        # Add num population in the top left of the frame (population = num frame_agents)
        cv2.putText(frame_img, f'Population: {len(frame_agents)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        video_writer.write(frame_img)
    video_writer.release()
