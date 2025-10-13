# Copyright (C) 2025 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Marius Kästingschäfer and Théo Gieruc
# ==============================================================================

import carla
import random
from common.vehicle import Vehicle


def spawn_cars(
    client,
    world,
    number_of_vehicles,
    blueprints,
    vehicle_transform,
    large_vehicles,
    sort_spawnpoints,
    traffic_manager,
    logger,
):
    '''
    Spawn vehicles in the Carla simulation world.
    
    Parameters:
        client (carla.Client): The Carla client object.
        world (carla.World): The Carla simulation world.
        number_of_vehicles (int): The number of vehicles to spawn.
        blueprints (list): The list of vehicle blueprints to choose from.
        vehicle_transform (carla.Transform): The transform of the vehicle to spawn around.
        large_vehicles (bool): Whether to spawn large vehicles or not.
        sort_spawnpoints (bool): Whether to sort the spawn points according to the distance to the vehicle_transform.
        logger (logging.Logger): The logger object to log messages.
        
    Returns:
        vehicles_list (list): The list of spawned vehicles.
    '''
    
    if number_of_vehicles <= 0:
        return []

    spawn_points = world.get_map().get_spawn_points()
    
    if sort_spawnpoints:
        # sort the spawn_points according to the distance to the vehicle_transform
        spawn_points = sorted(
            spawn_points, key=lambda x: x.location.distance(vehicle_transform.location)
        )[: number_of_vehicles * 2]
    else:
        random.shuffle(spawn_points)
    
    vehicles_list = []
    spawn_point_index = 0
    while len(vehicles_list) < number_of_vehicles:
        blueprint = random.choice(blueprints)
        large_vehicle_list = [
            "vehicle.carlamotors.carlacola",
            "vehicle.tesla.cybertruck",
            "vehicle.carlamotors.firetruck",
            "vehicle.mitsubishi.fusorosa",
            "vehicle.mercedes.sprinter",
            "vehicle.audi.etron",
            "vehicle.nissan.patrol_2021",
            "vehicle.volkswagen.t2_2021",
            "vehicle.jeep.wrangler_rubicon",
            "vehicle.volkswagen.t2",
            "vehicle.bmw.grandtourer",
            "vehicle.ford.ambulance",
            "vehicle.tesla.model3",
        ]
        medium_vehicle_list = [
            "vehicle.dodge.charger_police",
            "vehicle.dodge.charger_police_2020",
            "vehicle.chevrolet.impala",
            "vehicle.citroen.c3",
            "vehicle.toyota.prius",
            "vehicle.mini.cooper_s_2021",
            "vehicle.mercedes.coupe",
            "vehicle.lincoln.mkz_2020",
            "vehicle.seat.leon",
            "vehicle.nissan.micra",
            "vehicle.ford.crown",
            "vehicle.dodge.charger_2020",
            "vehicle.mercedes.coupe_2020",
            "vehicle.audi.a2",
        ]
        if large_vehicles == False:
            while blueprint.id in large_vehicle_list:
                blueprint = random.choice(blueprints)
        if blueprint.has_attribute("color"):
            color = random.choice(blueprint.get_attribute("color").recommended_values)

            blueprint.set_attribute("color", color)
        if blueprint.has_attribute("driver_id"):
            driver_id = random.choice(
                blueprint.get_attribute("driver_id").recommended_values
            )
            blueprint.set_attribute("driver_id", driver_id)
        blueprint.set_attribute("role_name", "autopilot")
        try:
            vehicle = Vehicle(blueprint, spawn_points[spawn_point_index], world, traffic_manager, logger)
            vehicles_list.append(vehicle)
        except RuntimeError as e:
            logger.info(e)
            #continue

        spawn_point_index += 1

    return vehicles_list


def spawn_pedestrians(client, world, number_of_walkers, blueprints_walkers, logger):
    '''
    Spawn pedestrians in the Carla simulation world.
    
    Parameters:
        client (carla.Client): The Carla client object. 
        world (carla.World): The Carla simulation world.
        number_of_walkers (int): The number of pedestrians to spawn.
        blueprints_walkers (list): The list of pedestrian blueprints to choose from.
        logger (logging.Logger): The logger object to log messages.
        
    Returns:
        walkers_list (list): The list of spawned pedestrians.
        all_id (list): The list of all spawned actors.
    
    '''
    
    if number_of_walkers <= 0:
        return

    spawn_points = []
    for _ in range(number_of_walkers):
        spawn_point = carla.Transform()
        loc = world.get_random_location_from_navigation()
        if loc is not None:
            spawn_point.location = loc
            spawn_points.append(spawn_point)

    batch = []
    walker_speed = []
    for spawn_point in spawn_points:
        walker_bp = random.choice(blueprints_walkers)
        if walker_bp.has_attribute("is_invincible"):
            walker_bp.set_attribute("is_invincible", "false")
        if walker_bp.has_attribute("speed"):
            walker_speed.append(walker_bp.get_attribute("speed").recommended_values[1])
        else:
            print("Walker has no speed")
            walker_speed.append(0.0)
        batch.append(carla.command.SpawnActor(walker_bp, spawn_point))

    results = client.apply_batch_sync(batch, True)
    walkers_list = []
    walker_speed2 = []
    for i in range(len(results)):
        if results[i].error:
            logger.info(results[i].error)
        else:
            walkers_list.append({"id": results[i].actor_id})
            walker_speed2.append(walker_speed[i])
    walker_speed = walker_speed2

    walker_controller_bp = world.get_blueprint_library().find("controller.ai.walker")
    batch = []
    for i in range(len(walkers_list)):
        batch.append(
            carla.command.SpawnActor(
                walker_controller_bp, carla.Transform(), walkers_list[i]["id"]
            )
        )

    results = client.apply_batch_sync(batch, True)
    for i in range(len(results)):
        if results[i].error:
            logger.info(results[i].error)
        else:
            walkers_list[i]["con"] = results[i].actor_id

    all_id = []
    for i in range(len(walkers_list)):
        all_id.append(walkers_list[i]["con"])
        all_id.append(walkers_list[i]["id"])

    all_actors = world.get_actors(all_id)
    world.tick()

    # Initialize each controller and set target to walk to
    for i in range(0, len(all_id), 2):
        all_actors[i].start()
        all_actors[i].go_to_location(world.get_random_location_from_navigation())
        all_actors[i].set_max_speed(float(walker_speed[int(i / 2)]))

    return walkers_list, all_id
