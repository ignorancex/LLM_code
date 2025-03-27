import carla
import random
import queue
import os 
import argparse
import numpy as np
import json
import time

index = 0

def rgb(image, PATH):
    global index
    image.save_to_disk(f"{PATH}/{int(index/2)}_rgb.png")
    index+=1

def seg_ground(image, PATH):
    global index
    image.save_to_disk(f"{PATH}/{int(index/2)}_seg_ground.png", carla.ColorConverter.CityScapesPalette)
    index+=1
    
def main():
    client = carla.Client('localhost', 2000)
    world = client.get_world()
    settings = world.get_settings()
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    spectator = world.get_spectator()
    
    #set weather using parameters from batch code
    weather_param,weather_path = parse_weather()
    world.set_weather(weather_param)
    
    #makes data directory
    PATH = f"Carla_data/Images/{weather_path}"
    if not os.path.exists(PATH): os.makedirs(PATH)
    else: return
         
    #find main car
    car = None
    for actor in world.get_actors():
        if(actor.attributes.get('role_name')=='hero'):
            car = actor
    
    #spawn rgb camera
    relative_transform = carla.Transform(carla.Location(0,0,2),carla.Rotation())
    rgb_cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    rgb_cam_bp.set_attribute("sensor_tick","5.0")
    rgb_cam_bp.set_attribute("image_size_x","1600")
    rgb_cam_bp.set_attribute("image_size_y","1200")
    rgb_camera = world.spawn_actor(rgb_cam_bp, relative_transform, attach_to = car)
    
    #spawn segmented ground truth camera
    segmented_cam_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
    segmented_cam_bp.set_attribute("sensor_tick","5.0")
    segmented_cam_bp.set_attribute("image_size_x","1600")
    segmented_cam_bp.set_attribute("image_size_y","1200")
    segmented_camera = world.spawn_actor(segmented_cam_bp, relative_transform, attach_to = car)
    
    rgb_camera.listen(lambda image: rgb(image, PATH))
    segmented_camera.listen(lambda image: seg_ground(image, PATH))
    
    #simulate
    while index<2000:
        location = car.get_transform().location
        spectator.set_transform(carla.Transform(carla.Location(location.x, location.y, location.z+2),car.get_transform().rotation))
        time.sleep(0.05)
        

def parse_weather():
    argparser = argparse.ArgumentParser(
        description="simulation innit all args between 0 and 100")
    argparser.add_argument(
        '--cloudiness',
        metavar='C',
        default=0.0,
        type = float,
        help='cloudiness')
    argparser.add_argument(
        '--precipitation',
        metavar='P',
        default=0.0,
        type=float,
        help='rain')
    argparser.add_argument(
        '--fog',
        metavar='F',
        default=0.0,
        type=float,
        help='fog')
    argparser.add_argument(
        '--dust',
        metavar='D',
        default=0.0,
        type=float,
        help='dust storm')
    argparser.add_argument(
        '--person',
        metavar='H',
        type=str,
        help='personid')
    
    args = argparser.parse_args()
    
    weather = carla.WeatherParameters(
        cloudiness=args.cloudiness,
        precipitation=args.precipitation,
        fog_density=args.fog,
        dust_storm=args.dust,
        sun_altitude_angle=90.0)
    path = f"{args.person}.{int(args.cloudiness)}.{int(args.precipitation)}.{int(args.fog)}"
    
    return weather,path

if __name__ == "__main__":
    main()
