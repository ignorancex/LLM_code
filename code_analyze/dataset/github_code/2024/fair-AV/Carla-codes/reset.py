import carla

def main():
    client = carla.Client('localhost', 2000)
    world = client.get_world()
    
    '''
    actors = world.get_actors()
    for actor in actors:
        actor.destroy()
    '''
    
    try:
        client.reload_world()
    except:
        pass
    
    print("Simulation Restarted")
    
if __name__ == "__main__":
    main()
    
    