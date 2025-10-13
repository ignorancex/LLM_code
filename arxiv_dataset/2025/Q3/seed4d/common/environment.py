# Copyright (C) 2025 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Marius Kästingschäfer and Théo Gieruc
# ==============================================================================

import carla


def init_world(town, delta_seconds, weather, unload):
    """
    Initializes a connection to the CARLA simulator and loads the specified town.

    Parameters:
        town (str): The name of the Carla town to load.
        delta_seconds (float): The fixed delta time elapsing per tick.
        weather (str): The weather condition to apply to the world.
        unload (bool): Used to untoggle all map elements.

    Returns:
        world (carla.World): The Carla simulation world.
        settings (carla.WorldSettings): The Carla simulation world settings.
        blueprint_library (carla.BlueprintLibrary): The Carla blueprint library (containing vehicle, camera, etc. blueprints).

    Raises:
        ValueError: If the specified weather condition is not recognized.
    """

    client = carla.Client("localhost", 2000)
    # this value might need to be adjusted, depending on the server speed
    # one wants to avoid long loading time, but the client (CARLA) needs to be ready when called!
    client.set_timeout(10.0) #10
    world = client.load_world(town)
    print("World loaded")

    settings = world.get_settings()
    settings.fixed_delta_seconds = delta_seconds
    settings.synchronous_mode = True
    world.apply_settings(settings)

    weather_options = {
        "ClearNoon": carla.WeatherParameters.ClearNoon,
        "CloudyNoon": carla.WeatherParameters.CloudyNoon,
        "WetNoon": carla.WeatherParameters.WetNoon,
        "WetCloudyNoon": carla.WeatherParameters.WetCloudyNoon,
        "MidRainyNoon": carla.WeatherParameters.MidRainyNoon,
        "HardRainNoon": carla.WeatherParameters.HardRainNoon,
        "SoftRainNoon": carla.WeatherParameters.SoftRainNoon,
        "ClearSunset": carla.WeatherParameters.ClearSunset,
        "CloudySunset": carla.WeatherParameters.CloudySunset,
        "WetSunset": carla.WeatherParameters.WetSunset,
        "WetCloudySunset": carla.WeatherParameters.WetCloudySunset,
        "MidRainSunset": carla.WeatherParameters.MidRainSunset,
        "HardRainSunset": carla.WeatherParameters.HardRainSunset,
        "SoftRainSunset": carla.WeatherParameters.SoftRainSunset,
    }

    if weather in weather_options:
        world.set_weather(weather_options[weather])
    else:
        print("Weather not found")

    if unload:
        world.unload_map_layer(carla.MapLayer.Buildings)
        world.unload_map_layer(carla.MapLayer.ParkedVehicles)
        world.unload_map_layer(carla.MapLayer.Ground)
        world.unload_map_layer(carla.MapLayer.Decals)
        world.unload_map_layer(carla.MapLayer.Foliage)
        world.unload_map_layer(carla.MapLayer.StreetLights)
        world.unload_map_layer(carla.MapLayer.Particles)
        world.unload_map_layer(carla.MapLayer.Walls)
        # world.unload_map_layer(carla.MapLayer.Props)
        # world.unload_map_layer(carla.MapLayer.All)

    settings = world.get_settings()
    blueprint_library = world.get_blueprint_library()

    return world, settings, blueprint_library
