import numpy as np
import ot
from tqdm import tqdm
from dtaidistance import dtw_ndim
from joblib import Parallel, delayed
from utils.utils import ProgressParallel
from utils.trajectory import image_to_world_trajectory


def measure_emd(source, target):
    """
    Measure the Earth Mover's Distance (EMD) between two sets of data points.

    Params:
        source (np.ndarray): Source data points with shape (..., N)
        target (np.ndarray): Target data points with shape (..., N)

    Returns:
        float: Earth Mover's Distance between the two sets of data points
    """
    
    assert source.ndim == target.ndim, 'Source and target data points must have the same number of dimensions'

    # Flatten the data points
    if source.ndim == 1:
        source_flat = source[:, np.newaxis]
        target_flat = target[:, np.newaxis]
    else:
        assert source.shape[-1] == target.shape[-1], 'Source and target data points must have the same size in the last dimension'
        source_flat = source.reshape(-1, source.shape[-1])
        target_flat = target.reshape(-1, target.shape[-1])

    # Calculate the pairwise cost matrix and uniform weight for each sample
    M = ot.dist(source_flat, target_flat, metric='euclidean')
    weights_source = np.ones(source_flat.shape[0]) / source_flat.shape[0]
    weights_target = np.ones(target_flat.shape[0]) / target_flat.shape[0]

    # Compute EMD
    emd = ot.emd2(weights_source, weights_target, M)
    return emd


def calculate_quadrat(scenario, scene_HW, num_quadrats=10, eval_every=25):
    """
    Calculate density, frequency (based on unique agent_type), and coverage of the scenario at every eval_every frames.

    Params:
        scenario (pd.DataFrame): Scenario data with columns=['scene', 'agent_id', 'agent_type', 'frame', 'x', 'y']
        scene_size (tuple): Size of the scene (height, width)
        num_quadrats (int): Number of quadrats per row/column (forming a num_quadrats x num_quadrats grid)
        eval_every (int): Number of frames to skip between evaluations

    Returns:
        np.ndarray: (T, 3) array where T is the number of frames evaluated
                    Columns represent density, frequency (unique agent types), and coverage at each evaluation time
    """

    # Initialize scene dimensions and quadrat sizes
    height, width = scene_HW
    quadrat_height = height / num_quadrats
    quadrat_width = width / num_quadrats

    # Downsample frames
    scenario = scenario[scenario['frame'] % eval_every == 0].copy()

    # Map agent coordinates to quadrat indices
    x_indices = (scenario['x'] // quadrat_width).astype(int).clip(0, num_quadrats - 1)
    y_indices = (scenario['y'] // quadrat_height).astype(int).clip(0, num_quadrats - 1)
    quadrat_indices = (y_indices * num_quadrats + x_indices).astype(int)
    scenario['quadrat_index'] = quadrat_indices

    results = []
    for frame, frame_data in scenario.groupby('frame'):
        # Density: Count agents in each quadrat
        quadrat_counts = np.bincount(frame_data['quadrat_index'], minlength=num_quadrats**2).reshape(num_quadrats, num_quadrats)
        density = quadrat_counts.sum() / (num_quadrats * num_quadrats)

        # Frequency: Count unique agent types in each quadrat
        unique_counts = frame_data.groupby('quadrat_index')['agent_type'].nunique().reindex(range(num_quadrats**2), fill_value=0)
        unique_counts = unique_counts.values.reshape(num_quadrats, num_quadrats)
        frequency = unique_counts.mean()

        # Coverage: Proportion of quadrats with at least one agent
        coverage = np.count_nonzero(quadrat_counts) / (num_quadrats * num_quadrats)

        # Append results for this frame
        results.append([density, frequency, coverage])

    return np.array(results)


def calculate_quadrat_fast(scenario, scene_HW, duration, num_quadrats=10, eval_every=25):
    """
    Calculate density, frequency (based on unique agent_type), and coverage of the scenario at every eval_every frames.

    Params:
        scenario (pd.DataFrame): Scenario data with columns=['scene', 'agent_id', 'agent_type', 'frame', 'x', 'y']
        scene_size (tuple): Size of the scene (height, width)
        num_quadrats (int): Number of quadrats per row/column (forming a num_quadrats x num_quadrats grid)
        eval_every (int): Number of frames to skip between evaluations

    Returns:
        np.ndarray: (T, 3) array where T is the number of frames evaluated
                    Columns represent density, frequency (unique agent types), and coverage at each evaluation time
    """

    # Initialize scene dimensions and quadrat sizes
    height, width = scene_HW
    quadrat_height = height / num_quadrats
    quadrat_width = width / num_quadrats

    # Downsample frames
    scenario = scenario[scenario['frame'] % eval_every == 0].copy()

    # Map agent coordinates to quadrat indices
    x_indices = (scenario['x'] // quadrat_width).astype(int).clip(0, num_quadrats - 1)
    y_indices = (scenario['y'] // quadrat_height).astype(int).clip(0, num_quadrats - 1)
    quadrat_indices = (y_indices * num_quadrats + x_indices).astype(int)
    scenario['quadrat_index'] = quadrat_indices

    # Group scenario data by frame and quadrat index
    grouped = scenario.groupby(['frame', 'quadrat_index'])

    # Density: Count agents in each quadrat per frame
    density_counts = grouped.size().unstack(fill_value=0)

    # Frequency: Count unique agent types in each quadrat per frame
    frequency_counts = grouped['agent_type'].nunique().unstack(fill_value=0)

    # Coverage: Proportion of occupied quadrats in each frame
    coverage_counts = (density_counts > 0).astype(int)

    # Calculate all metrics at once using vectorized operations
    num_quadrats_total = num_quadrats * num_quadrats
    density = density_counts.sum(axis=1) / num_quadrats_total
    frequency = frequency_counts.sum(axis=1) / num_quadrats_total
    coverage = coverage_counts.sum(axis=1) / num_quadrats_total

    # Fill in empty frames with zeros
    density = density.reindex(range(0, duration, eval_every), fill_value=0)
    frequency = frequency.reindex(range(0, duration, eval_every), fill_value=0)
    coverage = coverage.reindex(range(0, duration, eval_every), fill_value=0)

    results = np.column_stack((density.values, frequency.values, coverage.values))

    return results


def measure_quadrat_method_similarity(source, target, scene_HW, duration, fps, num_quadrats=10):
    """
    [Scene-Level Metrics]
    Calculate the Earth Mover's Distance (EMD) for density, frequency, and coverage using the quadrat method.
    Source and target scenarios should be in *pixel coordinates*.

    Params:
        source (pd.DataFrame): Source scenario data with columns=['scene', 'agent_id', 'agent_type', 'frame', 'x', 'y']
        target (pd.DataFrame): Target scenario data with columns=['scene', 'agent_id', 'agent_type', 'frame', 'x', 'y']
        scene_HW (tuple): Size of the scene (height, width)
        duration (int): Duration of the scenario in seconds
        fps (int): Frames per second of the scenario

    Returns:
        emd_density (float): Earth Mover's Distance for density
        emd_frequency (float): Earth Mover's Distance for frequency
        emd_coverage (float): Earth Mover's Distance for coverage
    """
    
    # Calculate the quadrat method metrics for source and target scenarios
    source_quadrat = calculate_quadrat_fast(source, scene_HW, duration, num_quadrats, eval_every=fps)  # Downsample to 1 second
    target_quadrat = calculate_quadrat_fast(target, scene_HW, duration, num_quadrats, eval_every=fps)  # Downsample to 1 second

    source_density, source_frequency, source_coverage = source_quadrat[:, 0], source_quadrat[:, 1], source_quadrat[:, 2]
    target_density, target_frequency, target_coverage = target_quadrat[:, 0], target_quadrat[:, 1], target_quadrat[:, 2]

    # Calculate the EMD between the source and target quadrat metrics
    emd_density = measure_emd(source_density, target_density)
    emd_frequency = measure_emd(source_frequency, target_frequency)
    emd_coverage = measure_emd(source_coverage, target_coverage)

    return emd_density, emd_frequency, emd_coverage


def measure_population_similarity(source, target, fps, norm=True):
    """
    [Scene-Level Metrics]
    Calculate the Earth Mover's Distance (EMD) for population similarity between source and target scenarios.
    
    Params:
        source (pd.DataFrame): Source scenario data with columns=['scene', 'agent_id', 'agent_type', 'frame', 'x', 'y']
        target (pd.DataFrame): Target scenario data with columns=['scene', 'agent_id', 'agent_type', 'frame', 'x', 'y']
        fps (int): Frames per second of the scenario
        norm (bool): Whether to normalize the population values by the target population mean (default=True)
        
    Returns:
        emd_population (float): Earth Mover's Distance for population similarity
    """
    
    # Calculate the population at each frame for source and target scenarios
    source_population = source[source['frame'] % fps == 0].groupby('frame').size().values  # Downsample to 1 second
    target_population = target[target['frame'] % fps == 0].groupby('frame').size().values  # Downsample to 1 second

    # Normalize the population values for better comparison across sparse and dense scenarios
    if norm:
        target_population_mean = target_population.mean()
        source_population = source_population / target_population_mean
        target_population = target_population / target_population_mean

    # Calculate the EMD between the source and target populations
    emd_population = measure_emd(source_population, target_population)

    return emd_population


def measure_appearance_density_similarity(source, target, fps):
    """
    [Scene-Level Metrics] [Not used]
    Calculate the Earth Mover's Distance (EMD) for appearance density similarity between source and target scenarios.
    Source and target scenarios should be in *meter coordinates*.

    Params:
        source (pd.DataFrame): Source scenario data with columns=['scene', 'agent_id', 'agent_type', 'frame', 'x', 'y']
        target (pd.DataFrame): Target scenario data with columns=['scene', 'agent_id', 'agent_type', 'frame', 'x', 'y']
        fps (int): Frames per second of the scenario

    Returns:
        emd_appearance_density (float): Earth Mover's Distance for appearance density similarity
    """
    
    # Calculate the origin and goal positions of agents in source and target scenarios
    source_origin_goal = source.groupby('agent_id').agg(origin_x=('x', 'first'), origin_y=('y', 'first'), goal_x=('x', 'last'), goal_y=('y', 'last')).to_numpy()
    target_origin_goal = target.groupby('agent_id').agg(origin_x=('x', 'first'), origin_y=('y', 'first'), goal_x=('x', 'last'), goal_y=('y', 'last')).to_numpy()

    # Calculate the EMD between the origin-goal positions
    emd_appearance_density = measure_emd(source_origin_goal, target_origin_goal)

    return emd_appearance_density


def measure_population_density_similarity(source, target, fps):
    """
    [Scene-Level Metrics] [Not used]
    Calculate the Earth Mover's Distance (EMD) for population density similarity between source and target scenarios.
    Source and target scenarios should be in *meter coordinates*.

    Params:
        source (pd.DataFrame): Source scenario data with columns=['scene', 'agent_id', 'agent_type', 'frame', 'x', 'y']
        target (pd.DataFrame): Target scenario data with columns=['scene', 'agent_id', 'agent_type', 'frame', 'x', 'y']
        fps (int): Frames per second of the scenario

    Returns:
        emd_population_density (float): Earth Mover's Distance for population density similarity
    """
    
    # Calculate the trajectories of agents in source and target scenarios
    source_traj_all = source[['x', 'y']].values  
    target_traj_all = target[['x', 'y']].values

    # Downsampling highly affects the result of appearance density and population density similarity
    # so we will not downsample the trajectories here.
    # source_traj_all = source[source['frame'] % fps == 0][['x', 'y']].values  # Downsample to 1 second
    # target_traj_all = target[target['frame'] % fps == 0][['x', 'y']].values  # Downsample to 1 second

    # Calculate the EMD between the trajectories
    emd_population_density = measure_emd(source_traj_all, target_traj_all)

    return emd_population_density


def measure_travel_distance_similarity(source, target, fps, norm=True):
    """
    [Agent-Level Metrics]
    Calculate the Earth Mover's Distance (EMD) for travel distance similarity between source and target scenarios.
    Source and target scenarios should be in *meter coordinates*.

    Params:
        source (pd.DataFrame): Source scenario data with columns=['scene', 'agent_id', 'agent_type', 'frame', 'x', 'y']
        target (pd.DataFrame): Target scenario data with columns=['scene', 'agent_id', 'agent_type', 'frame', 'x', 'y']
        fps (int): Frames per second of the scenario
        norm (bool): Whether to normalize the distance values by the target mean (default=True)

    Returns:
        emd_dist (float): Earth Mover's Distance for travel distance similarity
    """

    # Calculate the travel distance for each agent in source and target scenarios
    source, target = source.copy(), target.copy()
    source_dist = source.groupby('agent_id').apply(lambda x: np.linalg.norm(x[['x', 'y']].diff().dropna().values, axis=1).sum()).values
    target_dist = target.groupby('agent_id').apply(lambda x: np.linalg.norm(x[['x', 'y']].diff().dropna().values, axis=1).sum()).values

    # normalize the distance values
    if norm:
        target_dist_mean = target_dist.mean()
        source_dist = source_dist / target_dist_mean
        target_dist = target_dist / target_dist_mean

    # calculate the EMD between the two sets of travel distances
    emd_dist = measure_emd(source_dist, target_dist)

    return emd_dist


def measure_travel_velocity_similarity(source, target, fps, norm=True):
    """
    [Agent-Level Metrics]
    Calculate the Earth Mover's Distance (EMD) for travel velocity similarity between source and target scenarios.
    Source and target scenarios should be in *meter coordinates*.

    Params:
        source (pd.DataFrame): Source scenario data with columns=['scene', 'agent_id', 'agent_type', 'frame', 'x', 'y']
        target (pd.DataFrame): Target scenario data with columns=['scene', 'agent_id', 'agent_type', 'frame', 'x', 'y']
        fps (int): Frames per second of the scenario
        norm (bool): Whether to normalize the velocity values by the target mean (default=True)

    Returns:
        emd_vel (float): Earth Mover's Distance for travel velocity similarity
    """

    # Group by agent id and calculate the velocity through L2, then calculate the agent-wise mean of velocity.
    source, target = source.copy(), target.copy()
    source_vel = source.groupby('agent_id').apply(lambda x: np.linalg.norm(x[['x', 'y']].diff().dropna().values, axis=1).mean() * fps).values
    target_vel = target.groupby('agent_id').apply(lambda x: np.linalg.norm(x[['x', 'y']].diff().dropna().values, axis=1).mean() * fps).values

    # normalize the velocity values
    if norm:
        target_vel_mean = target_vel.mean()
        source_vel = source_vel / target_vel_mean
        target_vel = target_vel / target_vel_mean

    # calculate the EMD between the two sets of velocities
    emd_vel = measure_emd(source_vel, target_vel)

    return emd_vel


def measure_travel_acceleration_similarity(source, target, fps, norm=True):
    """
    [Agent-Level Metrics]
    Calculate the Earth Mover's Distance (EMD) for travel acceleration similarity between source and target scenarios.
    Source and target scenarios should be in *meter coordinates*.

    Params:
        source (pd.DataFrame): Source scenario data with columns=['scene', 'agent_id', 'agent_type', 'frame', 'x', 'y']
        target (pd.DataFrame): Target scenario data with columns=['scene', 'agent_id', 'agent_type', 'frame', 'x', 'y']
        fps (int): Frames per second of the scenario
        norm (bool): Whether to normalize the acceleration values by the target mean (default=True)

    Returns:
        emd_acc (float): Earth Mover's Distance for travel acceleration similarity
    """

    # Group by agent id and calculate the acceleration through L2, then calculate the agent-wise mean of acceleration.
    source, target = source.copy(), target.copy()
    source_acc = source.groupby('agent_id').apply(lambda x: np.linalg.norm(x[['x', 'y']].diff().diff().dropna().values, axis=1).mean() * fps).values
    target_acc = target.groupby('agent_id').apply(lambda x: np.linalg.norm(x[['x', 'y']].diff().diff().dropna().values, axis=1).mean() * fps).values

    # normalize the acceleration values
    if norm:
        target_acc_mean = target_acc.mean()
        source_acc = source_acc / target_acc_mean
        target_acc = target_acc / target_acc_mean

    # calculate the EMD between the two sets of accelerations
    emd_acc = measure_emd(source_acc, target_acc)

    return emd_acc


def measure_travel_time_similarity(source, target, fps, norm=True):
    """
    [Agent-Level Metrics]
    Calculate the Earth Mover's Distance (EMD) for travel time similarity between source and target scenarios.
    Source and target scenarios should be in *meter coordinates*.

    Params:
        source (pd.DataFrame): Source scenario data with columns=['scene', 'agent_id', 'agent_type', 'frame', 'x', 'y']
        target (pd.DataFrame): Target scenario data with columns=['scene', 'agent_id', 'agent_type', 'frame', 'x', 'y']
        fps (int): Frames per second of the scenario
        norm (bool): Whether to normalize the time values by the target mean (default=True)

    Returns:
        emd_time (float): Earth Mover's Distance for travel time similarity
    """

    # Calculate the travel time for each agent in source and target scenarios
    source, target = source.copy(), target.copy()
    source_time = source.groupby('agent_id').apply(lambda x: (x['frame'].iloc[-1] - x['frame'].iloc[0]) / fps).values
    target_time = target.groupby('agent_id').apply(lambda x: (x['frame'].iloc[-1] - x['frame'].iloc[0]) / fps).values

    # Normalize the travel duration values
    if norm:
        target_time_mean = target_time.mean()
        source_time = source_time / target_time_mean
        target_time = target_time / target_time_mean

    # Calculate the EMD between the two sets of travel durations
    emd_time = measure_emd(source_time, target_time)

    return emd_time


def measure_dtw_diversity(source, target, fps):
    """
    [Agent-Level Metrics]
    Calculate the minimum pairwise dynamic time warping distance and diversity between source and target scenarios.
    Source and target scenarios should be in *meter coordinates*.

    Params:
        source (pd.DataFrame): Source scenario data with columns=['scene', 'agent_id', 'agent_type', 'frame', 'x', 'y']
        target (pd.DataFrame): Target scenario data with columns=['scene', 'agent_id', 'agent_type', 'frame', 'x', 'y']

    Returns:
        dtw_distance (float): Minimum pairwise DTW distance between source and target scenarios
        dtw_diversity (float): Diversity of the trajectories based on DTW distance
    """

    source_traj = source.groupby('agent_id').apply(lambda x: x[['x', 'y']].values).values
    target_traj = target.groupby('agent_id').apply(lambda x: x[['x', 'y']].values).values

    # Make trajectories into double datatype to use distance_fast function
    source_traj = [traj.astype(np.double) for traj in source_traj]
    target_traj = [traj.astype(np.double) for traj in target_traj]

    do_parallel = len(source_traj) * len(target_traj) > 10000

    if not do_parallel:
        dtw_distance_matrix = np.zeros((len(source_traj), len(target_traj)))
        for i, s_traj in enumerate(source_traj):
            for j, t_traj in enumerate(target_traj):
                dtw_distance_matrix[i, j] = dtw_ndim.distance_fast(s_traj, t_traj)  # use_pruning=True (raise error), window=2 (makes worse)
    else:
        def compute_dtw_distance(i, s_traj):
            return [dtw_ndim.distance_fast(s_traj, t_traj) for t_traj in target_traj]
        
        dtw_distance_matrix = np.array(ProgressParallel(total=len(source_traj), n_jobs=-1)(delayed(compute_dtw_distance)(i, s_traj) for i, s_traj in enumerate(source_traj)))

    # Normalize by fps to provide a value independent to temporal resolution
    source_target_dtw_distance = dtw_distance_matrix.min(axis=1).mean() / fps
    target_source_dtw_distance = dtw_distance_matrix.min(axis=0).mean() / fps

    # Calculate the minimum pairwise DTW distance in both directions
    dtw_distance = (source_target_dtw_distance + target_source_dtw_distance) / 2

    # Calculate the diversity of the trajectory that indicates whether all trajectories are well covered
    source_target_dtw_coverage = np.zeros(dtw_distance_matrix.shape[1])
    source_target_dtw_coverage[dtw_distance_matrix.argmin(axis=1)] = 1
    source_target_dtw_coverage = source_target_dtw_coverage.mean()

    target_source_dtw_coverage = np.zeros(dtw_distance_matrix.shape[0])
    target_source_dtw_coverage[dtw_distance_matrix.argmin(axis=0)] = 1
    target_source_dtw_coverage = target_source_dtw_coverage.mean()

    dtw_diversity = (source_target_dtw_coverage + target_source_dtw_coverage) / 2

    return dtw_distance, dtw_diversity


def measure_collision_rate(source):
    """
    [Agent-Level Metrics]
    Calculate the collision rate of the generated scenario.
    Assumes that the collision is occurred when two agents are within 0.2 meters of each other at the same frame.
    Source scenario should be in *meter coordinates*.

    Params:
        source (pd.DataFrame): Source scenario data with columns=['scene', 'agent_id', 'agent_type', 'frame', 'x', 'y']
    Returns:
        source_collision_rate (float): Collision rate of the generated scenario
    """

    source_collision = 0
    for frame, frame_data in source.groupby('frame'):
        # Calculate the pairwise distance between agents at each frame
        x = frame_data[['x', 'y']].values
        dist = np.linalg.norm(x[:, None] - x, axis=-1)
        np.fill_diagonal(dist, np.inf)
        
        source_collision += np.count_nonzero((dist < 0.2).any(axis=0))  # 0.2 meters threshold

    source_collision_rate = source_collision / len(source)

    return source_collision_rate


def measure_origin_goal_joint_distance_error(source, target):
    """
    [Agent-Level Metrics] [Not used]
    Calculate the origin-goal joint distance error between source and target scenarios.
    Source and target scenarios should be in *meter coordinates*.
    Params:
        source (pd.DataFrame): Source scenario data with columns=['scene', 'agent_id', 'agent_type', 'frame', 'x', 'y']
        target (pd.DataFrame): Target scenario data with columns=['scene', 'agent_id', 'agent_type', 'frame', 'x', 'y']
    Returns:
        origin_goal_joint_distance_error (float): Minimum pairwise origin-goal distance between source and target scenarios
    """

    # Calculate the origin and goal positions of agents in source and target scenarios
    # dimension = (num_agents, (origin goal), (x, y))
    source_origin_goal = source.groupby('agent_id').agg(origin_x=('x', 'first'), origin_y=('y', 'first'), goal_x=('x', 'last'), goal_y=('y', 'last')).to_numpy().reshape(-1, 2, 2)
    target_origin_goal = target.groupby('agent_id').agg(origin_x=('x', 'first'), origin_y=('y', 'first'), goal_x=('x', 'last'), goal_y=('y', 'last')).to_numpy().reshape(-1, 2, 2)

    # Calculate the pairwise distance between origin and goal positions of agents
    origin_goal_distance_matrix = np.linalg.norm(source_origin_goal[None, :] - target_origin_goal[:, None], axis=-1).mean(axis=-1)
    source_target_min_distance = origin_goal_distance_matrix.min(axis=0).mean()
    target_source_min_distance = origin_goal_distance_matrix.min(axis=1).mean()
    origin_goal_joint_distance_error = (source_target_min_distance + target_source_min_distance) / 2

    return origin_goal_joint_distance_error


def compute_metrics(source, target, scene_info, H, verbose=False):
    """
    Evaluate the generated scenario against the ground truth scenario.
    
    params:
        source (pd.DataFrame): Generated scenario with columns=['scene', 'agent_id', 'agent_type', 'frame', 'x', 'y']
        target (pd.DataFrame): Ground truth scenario with columns=['scene', 'agent_id', 'agent_type', 'frame', 'x', 'y']
        scene_info (dict): Dictionary containing information about the scene
        H (np.ndarray): Homography matrix
        verbose (bool): Whether to print the evaluation results

    returns:
        out (dict): Evaluation metrics
    """

    assert len(source['scene'].unique()) == 1, 'Generated scenario should contain only one scene'
    assert len(target['scene'].unique()) == 1, 'Ground truth scenario should contain only one scene'
    
    # Extract the scene information
    scene = source['scene'].iloc[0]
    source = source.drop(columns='scene')
    target = target.drop(columns='scene')
    height, width = scene_info['height'], scene_info['width']
    duration, fps = scene_info['length'], scene_info['fps']
    print(f'Scene: {scene}') if verbose else None

    # Prepare the source and target scenarios in meter coordinates
    source_meter = image_to_world_trajectory(source, H)
    target_meter = image_to_world_trajectory(target, H)
    
    pbar = tqdm(range(14), desc=f'Evaluate scenario {scene}')


    ##############################
    #   Scene-Level Evaluation   #
    ##############################

    # Density, Frequency, Coverage (Quadrat Method)
    pbar.set_description(f'Evaluate scenario {scene}: Density (Dens.), Frequency (Freq.), Coverage (Cov.)')
    emd_density, emd_frequency, emd_coverage = measure_quadrat_method_similarity(source, target, (height, width), duration, fps)
    print(f'Density (Dens.): {emd_density}') if verbose else None
    print(f'Frequency (Freq.): {emd_frequency}') if verbose else None
    print(f'Coverage (Cov.): {emd_coverage}') if verbose else None
    pbar.update(3)

    # Population similarity
    pbar.set_description(f'Evaluate scenario {scene}: Population Similarity (Pop.)')
    emd_population = measure_population_similarity(source, target, fps)
    print(f'Population Similarity (Pop.): {emd_population}') if verbose else None
    pbar.update()

    # Appearance density similarity (Not used)
    pbar.set_description(f'Evaluate scenario {scene}: Appearance Density')
    emd_appearance_density = 0  # measure_appearance_density_similarity(source_meter, target_meter, fps)
    print(f'EMD Appearance Density: {emd_appearance_density}') if verbose else None
    pbar.update()

    # Population density similarity (Not used)
    pbar.set_description(f'Evaluate scenario {scene}: Population Density')
    emd_population_density = 0  # measure_population_density_similarity(source_meter, target_meter, fps)
    print(f'EMD Population Density: {emd_population_density}') if verbose else None
    pbar.update()
    

    ##############################
    #   Agent-Level Evaluation   #
    ##############################

    # Travel distance similarity
    pbar.set_description(f'Evaluate scenario {scene}: Travel distance')
    emd_dist = measure_travel_distance_similarity(source_meter, target_meter, fps)
    print(f'EMD Travel Distance: {emd_dist}') if verbose else None
    pbar.update()

    # Travel velocity similarity
    pbar.set_description(f'Evaluate scenario {scene}: Travel velocity')
    emd_vel = measure_travel_velocity_similarity(source_meter, target_meter, fps)
    print(f'EMD Travel Velocity: {emd_vel}') if verbose else None
    pbar.update()

    # Travel acceleration similarity
    pbar.set_description(f'Evaluate scenario {scene}: Travel acceleration')
    emd_acc = measure_travel_acceleration_similarity(source_meter, target_meter, fps)
    print(f'EMD Travel Acceleration: {emd_acc}') if verbose else None
    pbar.update()

    # Travel time similarity
    pbar.set_description(f'Evaluate scenario {scene}: Travel time')
    emd_time = measure_travel_time_similarity(source_meter, target_meter, fps)
    print(f'EMD Travel Time: {emd_time}') if verbose else None
    pbar.update()

    # Kinematics
    pbar.set_description(f'Evaluate scenario {scene}: Kinematics (Kinem.)')
    kinematics = (emd_vel + emd_acc + emd_dist + emd_time) / 4
    print(f'Kinematics (Kinem.): {kinematics}') if verbose else None
    pbar.update()

    # Minimum pairwise dynamic time warping & diversity
    pbar.set_description(f'Evaluate scenario {scene}: Minimum Pairwise Dynamic Time Warping (DTW), Diversity (Div.)')
    dtw_distance, dtw_diversity = measure_dtw_diversity(source_meter, target_meter, fps)
    print(f'Minimum Pairwise Dynamic Time Warping (DTW): {dtw_distance}') if verbose else None
    print(f'Diversity (Div.): {dtw_diversity}') if verbose else None
    pbar.update()

    # Collision rate
    pbar.set_description(f'Evaluate scenario {scene}: Collision Rate (Col.)')
    source_collision_rate = measure_collision_rate(source_meter)
    print(f'Collision Rate (Col.): {source_collision_rate}') if verbose else None
    pbar.update()

    # Origin-goal joint distance error (Not used)
    pbar.set_description(f'Evaluate scenario {scene}: Origin-Goal Joint Distance Error')
    # origin_goal_joint_distance = measure_origin_goal_joint_distance(source_meter, target_meter)
    origin_goal_joint_distance_error = 0
    print(f'Origin-Goal Joint Distance Error: {origin_goal_joint_distance_error}') if verbose else None
    pbar.update()


    # Evaluation complete
    pbar.set_description(f'Evaluate scenario {scene}')
    pbar.close()

    output = {
        'Scene-Level Realism Metrics': {
            'Density': emd_density,
            'Frequency': emd_frequency,
            'Coverage': emd_coverage,
            'Population': emd_population,
        },
        'Agent-Level Accuracy Metrics': {
            'Kineamtics': kinematics,
            'DTW': dtw_distance,
            'Diversity': dtw_diversity,
            'Collision': source_collision_rate,
        },
        'Detailed Metrics': {
            'Travel Velocity': emd_vel,
            'Travel Acceleration': emd_acc,
            'Travel Distance': emd_dist,
            'Travel Time': emd_time,
        },
        'Deprecated Metrics': {
            'Appearance Density': emd_appearance_density,
            'Population Density': emd_population_density,
            'Origin-Goal Joint Distance Error': origin_goal_joint_distance_error,
        }
    }

    return output
