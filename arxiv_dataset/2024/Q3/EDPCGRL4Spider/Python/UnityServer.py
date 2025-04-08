from time import sleep
from threading import Thread, Event
import zmq
from ManageCONST import readCONST

# Custom thread class for Unity server
class UnityServer(Thread):
    """Thread for managing communication with the Unity client."""
    
    def __init__(self, event):
        """Initialize the UnityServer with an event for stopping."""
        super(UnityServer, self).__init__()
        self.event = event
        context = zmq.Context()
        self.socket = context.socket(zmq.PUSH)
        self.socket.bind('tcp://*:5555')
    
    def run(self):
        """Run the server in a loop until the event is set."""
        import atexit
        atexit.register(self.exit_handler)
        
        while True:
            if self.event.is_set():
                self.stopUnity()
                sleep(1)
                self.socket.close()
                break
            
            self.sendSpiderAttr() 
            print('Unity thread running...')
        
        print('Unity closing down')

    def stopUnity(self):
        """Send a stop command to Unity."""
        data = {'str': 'stop'}
        self.socket.send_json(data)

    def sendSpiderAttr(self):
        """Send spider attributes to Unity."""
        sleep(1)
        print("Sending spider attributes!")
        CONST = readCONST()
        data = CONST["SpiderAttr"]
        self.socket.send_json(data)

    def changeSceneUnity(self, sceneName):
        """Change the scene in Unity."""
        print("Changing scene to:", sceneName)
        data = {'str': sceneName}
        self.socket.send_json(data)

    def exit_handler(self):
        """Clean up on exit."""
        print("My application is ending")
        self.socket.close()

# Example usage:
'''
# Create the event
event = Event()

# Create and start a new UnityServer thread
unity_thread = UnityServer(event)
unity_thread.start()

# Allow the thread to run for a while
sleep(30)

# Stop the worker thread
print('Main stopping thread')
event.set()

# Wait for the thread to finish
unity_thread.join()
'''
