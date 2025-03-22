
import json

class ConfigurationHolder(object):
    
    """ This class a holder of the configuration for any type of file.
    """
    
    # Initialization of an empty dictionary
    def __init__(self):
        
        self.config = {}
        
        return
        
    # Function to read the configuration from a python dictionary
    # Inputs:
    # - config: python dictionary containing the configuration
    def LoadConfigFromPythonDictionary(self, config):
        
        self.config = config
        
        return 
    
    # Function to read the configuration from a JSON file
    # Inputs:
    # - jsonFile: path to the JSON file where the configuration is stored
    def LoadConfigFromJSONFile(self, jsonFile):
        
        with open(jsonFile) as json_data_file:
            self.config = json.load(json_data_file)
        
        return