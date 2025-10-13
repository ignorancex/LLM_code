import json
from time import sleep
from threading import Event
from QLearning import QLearning
from Random import Random
from RuleBased import Rule_Based
import Logger
import ManageCONST
from EDAThread import EDAThread
from PredictionModelEDA import readBaseEDA

# Load constants from JSON configuration
with open('CONSTANTS.json') as json_file:
    CONST = json.load(json_file)

# Start logging program start time
Logger.logTime("Start Program")

# Create and start a new thread for EDA device
eventEDA = Event()
threadEDA = EDAThread(eventEDA)
threadEDA.start()
Logger.logTime("Start EDA")

# Uncomment if using UnityServer
# eventUnity = Event()
# threadUnity = UnityServer(eventUnity)
# threadUnity.start()
Logger.logTime("Start Unity Connection")

# Allow some time for the threads to initialize
sleep(5)

# Change the Unity scene to "Relaxing"
ManageCONST.writeUnityScene("Relaxing")
sleep(CONST["Relaxing_Duration"])  # Sleep for the duration specified in constants

# Read base EDA and write to manage constants
ManageCONST.writeBaseEDA(readBaseEDA())

# Execute methods in the specified order
order = CONST["orderEnv"]
for method in order:
    ManageCONST.writeUnityScene("Stressful")
    
    if method == "RL":
        Logger.logTime("Start RL")
        QLearning()
        Logger.logTime("End RL")
    else:
        Logger.logTime("Start RuleBased")
        Rule_Based()
        Logger.logTime("End RuleBased")
    
    print("End method!")
    
    # Return to "Relaxing" scene after each method
    ManageCONST.writeUnityScene("Relaxing")
    sleep(CONST["Relaxing_Duration"])  # Sleep for the duration specified in constants

# Allow some time before stopping threads
sleep(5)

# Stop the EDA thread
print('Stopping threads')
eventEDA.set()
# Uncomment if using UnityServer
# eventUnity.set()

# Wait for the EDA thread to finish
threadEDA.join()
