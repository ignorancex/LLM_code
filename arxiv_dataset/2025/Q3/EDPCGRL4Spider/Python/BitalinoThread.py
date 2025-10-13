from time import sleep
from threading import Thread, Event
import bitalino
from ManageCONST import readCONST
from DataManager import DataManager

class BitalinoThread(Thread):
    """Custom thread class for managing Bitalino data acquisition."""
    
    def __init__(self, event):
        """Initialize the Bitalino thread with configuration constants."""
        super().__init__()
        self.event = event
        self.device = None
        CONST = readCONST()
        
        # Configuration settings for Bitalino
        self.samplingRate = CONST["Bitalino"]["samplingRate"]
        self.macAddress = CONST["Bitalino"]["macAddress"]
        self.batteryThreshold = CONST["Bitalino"]["batteryThreshold"]
        self.counter = 0

        self.connectDevice()

    def run(self):
        """Run the thread task in a loop until the event is set."""
        while not self.event.is_set():
            self.readDataDevice()
        self.stopDevice()
        print('Bitalino closing down')

    def connectDevice(self):
        """Establish connection to the Bitalino device and start data acquisition."""
        acqChannels = [0, 1, 2, 3, 4, 5]

        # Connect to BITalino
        self.device = bitalino.BITalino(self.macAddress)

        # Set battery threshold
        self.device.battery(self.batteryThreshold)

        # Read BITalino version
        print(f"Bitalino version: {self.device.version()}")

        # Start data acquisition
        self.device.start(self.samplingRate, acqChannels)

    def stopDevice(self):
        """Stop data acquisition and close the connection to the Bitalino device."""
        self.device.stop()
        self.device.close()

    def readDataDevice(self):
        """Read data from the Bitalino device and save it using DataManager."""
        nSamples = 10
        runningTime = 1  # in seconds
        
        # List to hold PPG data
        PPG = []
        for _ in range(int(runningTime * self.samplingRate / nSamples)):
            data = self.device.read(nSamples)
            ppg = data[:, 5].tolist()
            PPG += ppg

        # Save the PPG data
        dataManager = DataManager()
        dataManager.save(self.counter, PPG)

        self.counter += 1

# Example usage of the BitalinoThread class
# event = Event()
# thread = BitalinoThread(event)
# thread.start()
# sleep(8)  # Allow the thread to run for a short period
# print('Main stopping thread')
# event.set()  # Signal the thread to stop
# thread.join()  # Wait for the thread to finish
