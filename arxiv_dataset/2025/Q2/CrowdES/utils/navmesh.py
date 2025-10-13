import math
import numpy as np
from typing import List, Tuple
from pathfinder import PathFinder, get_unit_vector
import pathfinder.pyrvo as rvo

from utils.homography import image2world, world2image


class PathFinderNew(PathFinder):
    """A pathfinder class to support shortest path finding and ORCA simulation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def search_shortest_path(self, start_points: Tuple[Tuple[float, float, float]], finish_points: Tuple[Tuple[float, float, float]]):
        # Search shortest path
        paths = []
        for start_point, finish_point in zip(start_points, finish_points):
            path = self.search_path(start_point, finish_point)
            paths.append(path)
        return paths
    
    def orca_simulation(self, start_points: Tuple[Tuple[float, float, float]], finish_points: Tuple[Tuple[float, float, float]], agent_radius: float, agent_speed: float, delta_time: float):
        # Add agents to the simulation
        for start_point, finish_point in zip(start_points, finish_points):
            agent_id = self.add_agent(start_point, agent_radius, agent_speed)
            self.set_agent_destination(agent_id, finish_point)
        # Simulation
        is_agents_arrived = [False] * len(start_points)
        agent_positions = {}
        while not all(is_agents_arrived):
            self.update(delta_time=delta_time)
            self._agents_id
            for agent_id in self._agents_id:
                agent_positions[agent_id] = self.get_agent_position(agent_id)

            agent_positions.append([self.get_agent_position(agent_index) for agent_index in range(len(start_points))])
            is_agents_arrived = [np.linalg.norm(np.array(agent_position, dtype=float) - np.array(finish_point, dtype=float)[[0, 2]]) < 1e-3 for agent_position, finish_point in zip(agent_positions[-1], finish_points)]
        return agent_positions

    def update(self, delta_time: float):
        # delete agents before update step
        if len(self._agents_to_delete) > 0:
            indexes_in_groups = []  # store here indexes in each group
            inner_indexes = []
            for group_index in range(self._groups_count):
                indexes_in_groups.append([])
            for agent_id in self._agents_to_delete:
                agent_inner_index = self._get_agent_inner_index(agent_id)
                inner_indexes.append(agent_inner_index)
                if agent_inner_index > -1:
                    group_index = self._agents_group[agent_inner_index]
                    # call delete from simulation
                    agent_in_group_index = self._get_agent_group_index(agent_id, self._agents_group_id[group_index])
                    indexes_in_groups[group_index].append(agent_in_group_index)
            for group_indexes in indexes_in_groups:
                group_indexes.sort()
                rvo.delete_agent(self._simulators[group_index], group_indexes)
            # and now we a ready to delete data from arrays
            inner_indexes.sort()
            for i in range(len(inner_indexes)):
                agent_inner_index = inner_indexes[len(inner_indexes) - 1 - i]
                agent_id = self._agents_id[agent_inner_index]
                group_index = self._agents_group[agent_inner_index]
                agent_in_group_index = self._get_agent_group_index(agent_id, self._agents_group_id[group_index])
                # delete
                self._agents_height.pop(agent_inner_index)
                self._agents_target_direction.pop(agent_inner_index)
                self._agents_target_index.pop(agent_inner_index)
                self._agents_targets.pop(agent_inner_index)
                self._agents_path.pop(agent_inner_index)
                self._agents_activity.pop(agent_inner_index)
                self._agents_speed.pop(agent_inner_index)
                self._agents_group_id[self._agents_group[agent_inner_index]].pop(agent_in_group_index)
                self._agents_group.pop(agent_inner_index)
                self._agents_id.pop(agent_inner_index)
            self._agents_to_delete = []

        for agent_inner_index, agent_id in enumerate(self._agents_id):
            should_deactivate: bool = False
            group_index = self._agents_group[agent_inner_index]
            sim = self._simulators[group_index]
            agent_index = self._get_agent_group_index(agent_id, self._agents_group_id[group_index])  # index of the agent in the simulator
            if self._agents_activity[agent_inner_index]:
                current_position = rvo.get_agent_position(sim, agent_index)
                # calculate velocity vector
                agent_target_index = self._agents_target_index[agent_inner_index]
                agent_targets_count = len(self._agents_targets[agent_inner_index])
                target = self._agents_targets[agent_inner_index][agent_target_index]  # get path for the agent and select proper position in the path
                to_vector = (target[0] - current_position[0], target[1] - current_position[1])
                distance_to_target = math.sqrt(to_vector[0]**2 + to_vector[1]**2)
                a_speed = self._agents_speed[agent_inner_index]
                # check is target is a finish point and agent close to it
                if agent_target_index == agent_targets_count - 1 and distance_to_target < delta_time * a_speed:
                    # set last velocity for the agent and deactivate it
                    a_velocity = get_unit_vector(to_vector)
                    last_speed = distance_to_target / delta_time
                    rvo.set_agent_pref_velocity(sim, agent_index, (a_velocity[0] * last_speed, a_velocity[1] * last_speed))
                    should_deactivate = True
                else:
                    local_dir = self._to_direction(current_position, target)
                    start_dir = self._agents_target_direction[agent_inner_index]
                    d = local_dir[0] * start_dir[0] + local_dir[1] * start_dir[1]
                    if d < 0.0:
                        # the agent go over the target
                        if agent_target_index  < agent_targets_count - 1:
                            # there are other targets in the path
                            # try to switch to the enxt target point
                            next_target: Tuple[float, float] = self._agents_targets[agent_inner_index][agent_target_index + 1]
                            is_next_visible: bool = rvo.query_visibility(sim, current_position, next_target)
                            if is_next_visible:
                                # the next target point is visible, switch to it
                                self._agents_target_index[agent_inner_index] += 1
                                # aslo update direction
                                self._agents_target_direction[agent_inner_index] = self._to_direction(target, next_target)
                                target = self._agents_targets[agent_inner_index][agent_target_index + 1]
                if self._agents_activity[agent_inner_index]:
                    if not self._continuous_moving and should_deactivate:
                        # stop calculating velocity for the agent
                        # in all other updates it will be set to zero
                        self._agents_activity[agent_inner_index] = False
                        # also clear the path
                        self._agents_path[agent_inner_index] = []
                    else:
                        # try to update the path
                        if len(self._agents_targets[agent_inner_index]) > 0:
                            target_position = self._agents_targets[agent_inner_index][-1]
                            # set the height of the start point the height of the start of the current segment
                            a_path = self.search_path((current_position[0], self._agents_height[agent_inner_index][self._agents_target_index[agent_inner_index]], current_position[1]), (target_position[0], self._agents_height[agent_inner_index][-1], target_position[1]))
                            self._set_agent_path(agent_id, a_path)  # set raw 3-float tuples path
                        if not should_deactivate:
                            to_vector = (target[0] - current_position[0], target[1] - current_position[1])
                            a_velocity = get_unit_vector(to_vector)
                            # set prefered velocity
                            rvo.set_agent_pref_velocity(sim, agent_index, (a_velocity[0] * a_speed, a_velocity[1] * a_speed))
                else:
                    rvo.set_agent_pref_velocity(sim, agent_index, (0.0, 0.0))
            else:
                rvo.set_agent_pref_velocity(sim, agent_index, (0.0, 0.0))
        # simulate in each group
        for sim in self._simulators:
            rvo.simulate(sim, delta_time, self._move_agents)

        if self._snap_to_navmesh and self._navmesh:
            for agent_index, agent_id in enumerate(self._agents_id):
                group_index = self._agents_group[agent_index]
                sim = self._simulators[group_index]
                agent_in_group_index = self._get_agent_group_index(agent_id, self._agents_group_id[group_index])
                agent_position = rvo.get_agent_position(sim, agent_in_group_index)  # return 2d-position
                # sample navmesh
                # before we should find proper y-coordinate of the agent
                # from rvo we obtain only 2d-position

                # TODO: may be we should store the proportion of the path segment for each agent
                # and then find the proper height by linear interpolation between height of the path segment endpoints

                # use height of the end point of the current segment in the path
                agent_heights = self._agents_height[agent_index]
                agent_target = self._agents_target_index[agent_index]
                if agent_target < len(agent_heights):
                    y_height = agent_heights[agent_target]
                else:
                    y_height = 0.0
                sample_position = self._navmesh.sample((agent_position[0], y_height, agent_position[1]))
                if sample_position:
                    # sample is not None
                    # set agent position
                    rvo.set_agent_position(sim, agent_in_group_index, (sample_position[0], sample_position[2]))


def shortest_pathfinder(vertices, polygons, start_points, finish_points):
    """
    Find the shortest path between start_points and finish_points on the navigation mesh.

    Params:
        vertices (list of tuple): List of vertices where each vertex is a tuple (x, y).
        polygons (list of list): List of polygons, where each polygon is a list of indices into the vertices list.
        start_points (list of tuple): List of start points where each point is a tuple (x, y).
        finish_points (list of tuple): List of finish points where each point is a tuple (x, y).

    Returns:
        paths (list of list): List of paths, where each path is a list of points in [x, y] format.
        If no path is found, the path will be a straight line from start to finish point.
    """

    # Convert them to unity coord system (x, 0, y)
    vertices_unity = [(x, 0, y) for x, y in vertices]
    start_points_unity = [(x, 0, y) for x, y in start_points]
    finish_points_unity = [(x, 0, y) for x, y in finish_points]

    # Shortest path finding
    pf = PathFinderNew(vertices_unity, polygons)
    paths = pf.search_shortest_path(start_points_unity, finish_points_unity)

    # if path is empty list, then replace it to straight path to goal
    for i, path in enumerate(paths):
        if len(path) == 0:
            paths[i] = [start_points[i], finish_points[i]]
        else:
            paths[i] = [[x, y] for x, _, y in path]
    
    return paths


def locate_point_on_path(paths, distances):
    """
    Calculates the point on each path at a specified distance from the start.
    
    Params:
        paths (list of list): A list of paths, where each path is a list of points in [x, y] format.
        distances (list of float): A list of distances in meters from the start of each path.
    
    Returns:
        points (list of list): A list of [x, y] points at the specified distances on each path. 
        If the distance exceeds the length of a path, the last point of the path is returned.
    """

    result_points = []
    for path, distance in zip(paths, distances):
        path = np.array(path)

        if len(path) == 0:
            raise ValueError('Path is empty')
        elif len(path) == 1:
            # start and end point are the same
            result_points.append(path[0])
            continue

        segment_vectors = np.diff(path, axis=0)
        
        # Calculate segment lengths
        segment_lengths = np.linalg.norm(segment_vectors, axis=1)
        cumulative_lengths = np.cumsum(segment_lengths)
                    
        # Check if the distance is within the path length
        if distance <= cumulative_lengths[-1]:
            # Find the segment where the distance falls
            segment_index = np.searchsorted(cumulative_lengths, distance)
            
            # Calculate the start point, direction vector and the distance for the target segment
            segment_start = path[segment_index]
            segment_direction = segment_vectors[segment_index]
            distance_from_segment_start = distance - (cumulative_lengths[segment_index - 1] if segment_index > 0 else 0)
            
            # Normalize the direction vector and find the target point
            if segment_lengths[segment_index] < 1e-6:
                target_point = segment_start # Avoid division by zero
            else:
                segment_unit_vector = segment_direction / segment_lengths[segment_index]
                target_point = segment_start + distance_from_segment_start * segment_unit_vector
            result_points.append(target_point.tolist())
        else:
            # If the distance exceeds the path length, return the last point
            # result_points.append(path[-1].tolist())

            # Version 2 
            # If the distance exceeds the path length, return the extrapolated point on the last segment
            # because agents too slow to reach the goal at the end of the path
            segment_start = path[-2]
            segment_end = path[-1]
            segment_vector = segment_end - segment_start
            segment_length = np.linalg.norm(segment_vector)

            if segment_length < 1e-4:
                result_points.append(segment_end.tolist())
            else:
                segment_unit_vector = segment_vector / segment_length
                remaining_distance = distance - cumulative_lengths[-1]
                target_point = segment_end + remaining_distance * segment_unit_vector
                result_points.append(target_point.tolist())

    return result_points


def get_control_point(vertices, polygons, start_points, finish_points, homography, distances):
    """
    Get control points on the shortest path from start_points to finish_points on the navigation mesh.
    
    Params:
        vertices (list of tuple): List of vertices where each vertex is a tuple (x, y).
        polygons (list of list): List of polygons, where each polygon is a list of indices into the vertices list.
        start_points (list of tuple): List of start points where each point is a tuple (x, y).
        finish_points (list of tuple): List of finish points where each point is a tuple (x, y).
        homography (numpy.ndarray): Homography matrix to convert pixel coordinates to world coordinates.
        distances (list of float): List of distances in meters from the start of each path.

    Returns:
        control_points (list of list): List of control points in world coordinates for each path.
    """

    # Get shortest path
    paths_pixel = shortest_pathfinder(vertices, polygons, start_points, finish_points)  # pixel coordinates

    # Get control points in meter
    paths_meter = [image2world(np.array(path), homography) for path in paths_pixel]  # meter coordinates
    control_points = locate_point_on_path(paths_meter, distances)

    return control_points


def are_points_collinear(points: List[Tuple[float, float, float]]) -> bool:
    """
    Check if a list of points are collinear in 3D space.

    Params:
        points (list of tuple): List of points where each point is a tuple (x, y, z).

    Returns:
        out (bool): True if the points are collinear, False otherwise.
    """

    # At least 3 points are needed for collinearity check
    if len(points) < 3:
        return False
    
    # Compute the norm of the direction vector from the first two points
    points = np.array(points)
    direction = points[1] - points[0]
    norm_direction = np.linalg.norm(direction)
    
    # If the direction vector is zero (the points are identical), check all are identical
    if norm_direction == 0:
        return np.allclose(points, points[0])
    
    direction = direction / norm_direction
    
    # Check collinearity for each point using vector projections
    for i in range(2, len(points)):
        vector = points[i] - points[0]
        projection_error = vector - np.dot(vector, direction) * direction
        
        if not np.allclose(projection_error, 0):
            return False
    
    return True


def filter_collinear_polygons(vertices, polygons):
    """
    Filter out collinear polygons from a list of polygons.
    
    Params:
        vertices (list of tuple): List of vertices where each vertex is a tuple (x, y, z).
        polygons (list of list): List of polygons, where each polygon is a list of indices into the vertices list.
    Returns:
        filtered_polygons (list of list): List of polygons that are not collinear.
    """
    
    # Check if the points in each polygon are collinear
    filtered_polygons = []
    for polygon in polygons:
        points = [vertices[i] for i in polygon]
        if not are_points_collinear(points):
            filtered_polygons.append(polygon)
    return filtered_polygons
