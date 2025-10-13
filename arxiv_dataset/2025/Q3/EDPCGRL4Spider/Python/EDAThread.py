from time import sleep
from threading import Thread, Event
import bitalino
from ManageCONST import readCONST
from statistics import mean
from PIL import ImageGrab
from DataManager import DataManager

"""
Due to licensing issues associated with direct access to EDA signals, 
we have implemented an alternative approach. Specifically, we capture 
the EDA signal from the monitor screen using a technique that allows 
us to effectively read the data without direct access.
"""

class EDAThread(Thread):
    """Custom thread for handling EDA data collection."""
    
    def __init__(self, event):
        """Initialize the EDA thread with configuration constants."""
        super().__init__()
        self.event = event
        CONST = readCONST()
        
        # Extract configuration settings for EDA
        self.indexColorRange = (
            CONST["EDA"]["index_color_min"],
            CONST["EDA"]["index_color_max"]
        )
        self.subjectColorRange = (
            CONST["EDA"]["subject_color_min"],
            CONST["EDA"]["subject_color_max"]
        )
        self.frameSize = CONST["EDA"]["frame_size"]
        self.minScale = CONST["EDA"]["min_scale"]
        self.maxScale = CONST["EDA"]["max_scale"]

        # Calculate initial scales and prepare data manager
        self.scales = self.calculateScale()
        self.counter = 0
        self.dataManager = DataManager()

    def run(self):
        """Run the thread task in a loop until the event is set."""
        while not self.event.is_set():
            sleep(1)  # Sleep for a second
            self.repeatedReading()

        print('EDA thread is closing down')

    def getScale(self, pixels, colorRange):
        """Retrieve scale positions based on color range from the image pixels."""
        positions = []
        for x in range(self.frameSize[2] - self.frameSize[0]):
            for y in range(self.frameSize[3] - self.frameSize[1]):
                pix = pixels[x, y]
                if all(colorRange[0][z] <= pix[z] <= colorRange[1][z] for z in range(3)):
                    positions.append(y)
        return positions if positions else None

    def getY(self, pixels, colorRange):
        """Get the average Y position of pixels within the color range."""
        positions = self.getScale(pixels, colorRange)
        return mean(positions) if positions else None

    def calculateScale(self):
        """Calculate scales based on the initial frame capture."""
        scales = None
        while scales is None:
            pixels = ImageGrab.grab(bbox=self.frameSize).load()
            scales = self.getScale(pixels, self.indexColorRange)
            print(scales)  # Debugging output for scales

        eachScale = (self.maxScale - self.minScale) / (len(scales) - 1)
        print(eachScale)  # Debugging output for scale increment
        return scales

    def repeatedReading(self):
        """Perform repeated reading of EDA data."""
        y = None
        pixels = ImageGrab.grab(bbox=self.frameSize).load()
        
        while y is None:
            y = self.getY(pixels, self.subjectColorRange)  
        
        # Scale the Y value to fit within the defined scale
        yScaled = (y - self.scales[0]) * (self.maxScale - self.minScale) / (self.scales[-1] - self.scales[0]) 
        yScaled = self.maxScale - self.minScale - yScaled
        print(yScaled)  # Debugging output for scaled Y value
        
        # Save the scaled value using DataManager
        self.dataManager.save(self.counter, yScaled)
        self.counter += 1
        return yScaled

# Example usage of the EDAThread class
# event = Event()
# thread = EDAThread(event)
# thread.start()
# sleep(10)  # Allow the thread to run for a short period
# thread.dataManager.loadAll()  # Load all data after collection
# print('Main stopping thread')
# event.set()  # Signal the thread to stop
# thread.join()  # Wait for the thread to finish
