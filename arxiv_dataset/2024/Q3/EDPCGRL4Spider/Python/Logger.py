from ManageCONST import readCONST
from datetime import datetime

def logStr(message):
    """Append a message to the log file for the current subject."""
    CONST = readCONST()
    file_name = f"Logs/log{CONST['SubjectID']}"
    
    with open(file_name, "a+") as file_object:
        file_object.write(message + "\n")

def logTime(message):
    """Log a message with the current time."""
    now = datetime.now().time()  # Get the current time
    logStr(f"{message} > {now}")

def log(key, val):
    """Log a key-value pair with a timestamp."""
    message = f"{key} : {val}"
    logTime(message)
