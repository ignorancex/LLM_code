#!/usr/bin/env python3

import serial
from datetime import datetime
import os

arduino = serial.Serial("COM7", 115200, timeout=5)

def get_filename():
    now = datetime.now()
    now_str = now.strftime("%d-%m-%Y-%H-%M-%S")
    filename = os.getcwd() + "/data/" + now_str + ".txt"
    return filename

f = open(get_filename(), 'w')
n_lines = 0
arduino.read_all() # Clear the buffer before we start logging
while True:
    # Get the string
    b_newline = arduino.readline()
    str_newline = b_newline.decode()
    
    # Check if it's a signal for a new file
    if "Logging start" in str_newline:
        # Close the previous file
        f.close()
        if n_lines < 100:
            # Remove files that are too short.
            # This is because during testing, if the digital pin is unplugged, it''ll
            # rapidly switch between on-off, which will create a lot of junk files.
            os.remove(f.name)

        # Create a new file
        f = open(get_filename(), 'w')
        n_lines = 0

        # Units are in Time(ms) and Pressure(psi).
        str_newline = "Time,Pressure"
        
    print(str_newline, end="")
    n_lines += 1

    f.write(str_newline)
