#!/usr/bin/env python

import time
import rospy
import rospkg
import numpy as np
import copy
import cv2
import pyrealsense2 as rs
import os
import yaml

from modules.mapmanager import Polygon3D, MapManager
from modules.image_processes_cupy import process_polygon, project_points_to_plane, depth_to_point_cloud_region, fit_plane_ransac, anisotropic_diffusion, depth_to_normals, add_border_conditionally
from modules.ros_node import MappingNode

def nothing(x):
    """Placeholder function for trackbar."""
    pass

def load_config(config_file):
    """Load configuration from a YAML file."""
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)  
    return config

def create_dataset_folder(base_path):
    """Create a unique dataset folder and subfolders for RGB and depth images."""
    dataset_index = 1
    while True:
        dataset_name = f"dataset_{dataset_index}"
        dataset_path = os.path.join(base_path, dataset_name)
        if not os.path.exists(dataset_path):
            os.makedirs(os.path.join(dataset_path, "rgb"), exist_ok=True)
            os.makedirs(os.path.join(dataset_path, "depth"), exist_ok=True)
            return dataset_path
        dataset_index += 1

def set_cv_config(thresh1=50, thresh2=100, num_iter=60, kappa=134, gamma=2):
    """Initialize OpenCV windows and trackbars for image processing parameters."""
    cv2.namedWindow('Filtered Image with Largest Contour')
    cv2.createTrackbar('Canny thresh1', 'Filtered Image with Largest Contour', thresh1, max(thresh1, 1000), nothing)
    cv2.createTrackbar('Canny thresh2', 'Filtered Image with Largest Contour', thresh2, max(thresh2, 1000), nothing)

    cv2.namedWindow('smoothed_depth_img')
    cv2.createTrackbar('num_iter', 'smoothed_depth_img', num_iter, max(num_iter, 360), nothing)
    cv2.createTrackbar('kappa', 'smoothed_depth_img', kappa, max(kappa, 500), nothing)
    cv2.createTrackbar('gamma', 'smoothed_depth_img', gamma, max(gamma, 80), nothing)

def main():
    # Initialize ROS package manager
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('polygon_mapping')  # Get package path
    config_path = os.path.join(package_path, 'config', 'config_param.yaml')  # Construct path to config file
    config_param = load_config(config_path)  # Load configuration

    try:
        # Initialize map manager with z-axis merging threshold
        map_manager = MapManager(z_threshold=config_param['merge_dis_thresh'])
        node = MappingNode(map_manager, map_frame=config_param['map_frame'], camera_frame=config_param['camera_frame'])
        time.sleep(1)  # Wait 3 seconds for listener setup

        # Initialize and configure RealSense L515
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(config_param['camera_serial_id'])
        device = config.resolve(pipeline).get_device()
        depth_sensor = device.first_depth_sensor()

        # Set depth sensor options
        preset_index = 0  
        depth_sensor.set_option(rs.option.visual_preset, preset_index)

        # Enable color and depth streams
        config.enable_stream(rs.stream.color, config_param['rgb_width'], config_param['rgb_height'], rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, config_param['depth_width'], config_param['depth_height'], rs.format.z16, 30)
        fx, fy, cx, cy = config_param['depth_fx'], config_param['depth_fy'], config_param['depth_cx'], config_param['depth_cy']

        # Start pipeline streaming
        pipeline.start(config)

        # Set additional RealSense options if supported
        if depth_sensor.supports(rs.option.pre_processing_sharpening):
            depth_sensor.set_option(rs.option.pre_processing_sharpening, config_param['L515_pre_processing_sharpening'])
            print("Pre-processing sharpening set to ", config_param['L515_pre_processing_sharpening'])

        if depth_sensor.supports(rs.option.laser_power):
            depth_sensor.set_option(rs.option.laser_power, config_param['L515_laser_power'])
            print("Laser power set to ", config_param['L515_laser_power'])

        if depth_sensor.supports(rs.option.confidence_threshold):
            depth_sensor.set_option(rs.option.confidence_threshold, config_param['L515_confidence_threshold'])
            print("Confidence threshold set to ", config_param['L515_confidence_threshold'])

        # Set up OpenCV configurations
        set_cv_config(config_param['Canny_thresh1'], config_param['Canny_thresh2'], config_param['anisotropic_diffusion_num_iter'], config_param['anisotropic_diffusion_kappa'], config_param['anisotropic_diffusion_gamma'])
        closing_kernel = config_param['closing_kernel']

        # Prepare dataset folder and file paths if data saving is enabled
        if config_param['save_data']:
            image_index = 1
            dataset_root = os.path.join(package_path, 'data')
            dataset_path = create_dataset_folder(dataset_root)
            os.makedirs(os.path.join(dataset_path, "rgb"), exist_ok=True)
            os.makedirs(os.path.join(dataset_path, "depth"), exist_ok=True)
            t_file_path = os.path.join(dataset_path, "transformation_matrix.txt")

        # Main loop 
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            if not depth_frame:
                continue

            time_start_loop = time.time()
            depth_image = np.asanyarray(depth_frame.get_data())
            timestamp = time.time()

            T_lock = copy.deepcopy(node.T)  # Capture transformation matrix

            # Apply anisotropic diffusion to smooth depth image
            time_start = time.time()
            smoothed_depth_img = anisotropic_diffusion(depth_image)
            print('Anisotropic diffusion time cost', 1000 * (time.time() - time_start), 'ms')

            # Compute surface normals from the depth image
            normals = depth_to_normals(smoothed_depth_img, fx, fy, cx, cy)

            # Retrieve Canny threshold values from trackbars
            thresh1 = cv2.getTrackbarPos('Canny thresh1', 'Filtered Image with Largest Contour')
            thresh2 = cv2.getTrackbarPos('Canny thresh2', 'Filtered Image with Largest Contour')

            # Apply sharpening filter to enhance edges if enabled
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            if config_param['use_sharpening']:
                filtered_img = cv2.filter2D(normals, -1, kernel)
            else:
                filtered_img = normals

            # Add border to the filtered image
            filtered_img = add_border_conditionally(filtered_img)
            # Perform edge detection
            edges = cv2.Canny(filtered_img, thresh1, thresh2)

            # Use morphological closing to fill gaps in edges
            kernel = np.ones((closing_kernel, closing_kernel), np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            # Find contours in the edge-detected image
            contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            min_area_thresh = config_param['min_area_thresh']
            new_polygon_list = []

            for i, contour in enumerate(contours):
                # Skip contours without parent contours
                if hierarchy[0][i][3] == -1:
                    continue

                # Skip contours below area threshold
                if cv2.contourArea(contour) < min_area_thresh:
                    continue

                # Simplify contours using polygon approximation
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if cv2.contourArea(approx) < min_area_thresh:
                    continue

                # Draw contour and vertices
                cv2.drawContours(filtered_img, [approx], 0, (0, 255, 0), 2)
                for point in approx:
                    cv2.circle(filtered_img, tuple(point[0]), 5, (0, 0, 255), -1)

                polygon_points = depth_to_point_cloud_region(smoothed_depth_img, approx.reshape(-1, 2), fx, fy, cx, cy)
                if polygon_points.size == 0:
                    print("No valid points in the polygon region")
                    continue

                try:
                    # Fit a plane using RANSAC
                    time_start = time.time()
                    normal, d = fit_plane_ransac(
                        polygon_points,
                        max_trials=config_param['ran_max_trials'],
                        min_samples=config_param['ran_min_samples'],
                        residual_threshold=config_param['ran_residual_threshold'],
                        outlier_threshold=config_param['ran_outlier_threshold']
                    )
                    print('fit_plane_ransac time cost', 1000 * (time.time() - time_start), 'ms')

                    projected_points = project_points_to_plane((*normal, d), fx, fy, cx, cy, approx.reshape(-1, 2))

                    # Process polygon and transform to world coordinates
                    angle, world_vertices = process_polygon(projected_points, normal, T_lock)
                    if world_vertices is not None:
                        new_polygon_list.append(world_vertices)

                except ValueError as e:
                    print("Error in fitting plane:", e)

            # Add polygons to the map manager and estimate z-drift
            map_manager.add_polygon_list(new_polygon_list)

            # Display the polygon map
            if map_manager.polygons:
                map_img = map_manager.plot_polygons()
                cv2.imshow('Polygon Map', map_img)

            # Save images and transformation matrix if data saving is enabled
            if config_param['save_data']:
                rgb_filename = os.path.join(dataset_path, "rgb", f"{image_index:06d}.png")
                cv2.imwrite(rgb_filename, color_image)

                depth_filename = os.path.join(dataset_path, "depth", f"{image_index:06d}.png")
                cv2.imwrite(depth_filename, depth_image)

                with open(t_file_path, 'a') as t_file:
                    t_file.write(f"{image_index},{T_lock.flatten()},{timestamp}\n")
                image_index += 1

            # Display results
            cv2.imshow('smoothed_depth_img', cv2.applyColorMap(cv2.convertScaleAbs(smoothed_depth_img, alpha=0.03), cv2.COLORMAP_JET))
            cv2.imshow('normals Image', normals)
            cv2.imshow('Color Image', color_image)
            cv2.imshow('Edges', edges)
            cv2.imshow('Filtered Image with Largest Contour', filtered_img)
            print('Loop time cost', 1000 * (time.time() - time_start_loop), 'ms')

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        rospy.spin()

    except rospy.ROSInterruptException:
        pass

    finally:
        # Ensure pipeline stops and all OpenCV windows close
        pipeline.stop()
        cv2.destroyAllWindows()

        if config_param.get('save_data', False):
            localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            map_manager.save_polygons_to_file(os.path.join(dataset_path, f'{localtime}_polygons.txt'))
            cv2.imwrite(os.path.join(dataset_path, f'{localtime}_map_img.jpg'), map_img)

if __name__ == '__main__':
    main()
