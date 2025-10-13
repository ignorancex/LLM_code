import ManageCONST as MC

def writeInFile(data):
    """Write key-value pairs from a dictionary to a file."""
    CONST = MC.readCONST()
    path = CONST["PathUnity"]
    while True:
        try:
            with open(path, "w") as f:
                for key, val in data.items():
                    f.write(f"{key}:{val}\n")
            break  # Exit the loop if write is successful
        except Exception as e:
            print(f"Error writing to file: {e}")
            # Consider adding a delay or a retry limit here if necessary

def changeSceneUnity(sceneName):
    """Change the Unity scene by writing the scene name to the file."""
    print("Changing scene to:", sceneName)
    data = {'str': sceneName}
    writeInFile(data)

def writeSpiderAttrUnity():
    """Write Spider attributes to the file."""
    CONST = MC.readCONST()
    data = CONST["SpiderAttr"]
    writeInFile(data)
