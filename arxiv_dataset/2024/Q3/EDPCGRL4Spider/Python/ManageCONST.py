import json
import ConnectUnity

def readCONST():
    """Read and return constants from the JSON file."""
    with open('CONSTANTS.json') as json_file:
        CONST = json.load(json_file)
    return CONST

def writeUnityScene(sceneName):
    """Update the Unity scene in the constants and notify Unity."""
    with open('CONSTANTS.json', "r+") as json_file:
        CONST = json.load(json_file)
        CONST["SpiderAttr"]["str"] = sceneName
        json_file.seek(0)
        json.dump(CONST, json_file, indent=4)
        json_file.truncate()
    
    ConnectUnity.changeSceneUnity(sceneName)

def writeSpiderAttrDict(attributes_dict):
    """Update Spider attributes from a dictionary."""
    with open('CONSTANTS.json', "r+") as json_file:
        CONST = json.load(json_file)
        CONST["SpiderAttr"].update(attributes_dict)
        json_file.seek(0)
        json.dump(CONST, json_file, indent=4)
        json_file.truncate()

def writeSpiderAttr(attributes):
    """Convert a list of attributes and update the Spider attributes."""
    attributes = attributes.tolist()
    attr = {
        "Locomotion": attributes[0],
        "Motion": attributes[1],
        "Closeness": attributes[2],
        "Largness": attributes[3],
        "Hairiness": attributes[4],
        "Color": attributes[5]
    }
    writeSpiderAttrDict(attr)
    ConnectUnity.writeSpiderAttrUnity()

def writeTargetStressLevel(stresslevel):
    """Update the target stress level in the constants."""
    with open('CONSTANTS.json', "r+") as json_file:
        CONST = json.load(json_file)
        CONST["TargetSTRESS"] = stresslevel
        json_file.seek(0)
        json.dump(CONST, json_file, indent=4)
        json_file.truncate()

def writeBaseEDA(baseEDA):
    """Update the base EDA value in the constants."""
    print("Base EDA: ", baseEDA)
    with open('CONSTANTS.json', "r+") as json_file:
        CONST = json.load(json_file)
        CONST["EDA"]["base_EDA"] = baseEDA
        json_file.seek(0)
        json.dump(CONST, json_file, indent=4)
        json_file.truncate()
