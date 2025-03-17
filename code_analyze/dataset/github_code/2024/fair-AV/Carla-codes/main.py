import subprocess
import sys
import os
import time
import argparse

# person = ['0001'] #,'0024','0036'] #'0004','0011',
# weather = ['fog'] 

def main():

    parser = argparse.ArgumentParser(description='Generating imgs for carla')
    parser.add_argument('--person', type=str, default="0001", help='person id')
    parser.add_argument('--weather', type=str, default="fog", help='weather')
    parser.add_argument('--intensity', type=int, default=0, help='intensity')
    args = parser.parse_args()


    with open(os.devnull, 'w') as null_file:
        
        start = time.time()
        
        #traffic manager and main simulation
        proc1 = subprocess.Popen([sys.executable,'Carla-codes/generate_traffic_no_tick.py','--seed','69','--seedw','69',"--car-lights-on",'--number-of-walkers','40','--number-of-vehicles','1','--hero','--filterw',f'walker.pedestrian.{args.person}'], stdout=null_file)
        time.sleep(10)
        proc2 = subprocess.Popen([sys.executable,'Carla-codes/simulate.py',f'--{args.weather}',f'{args.intensity}','--person',args.person]) 

        #reset the simulation when done
        proc2.wait()
        proc1.terminate()
        proc3 = subprocess.Popen([sys.executable,'Carla-codes/reset.py',], stdout=null_file) 
        proc3.wait()
        time.sleep(20)
        print(f"done with simulation {args.person} {args.weather} {args.intensity} at time {time.time()-start}")
    
if __name__ == "__main__":
    main()