import json
from pathlib import Path


class Save:
    def __init__(self, path, should_exist=False):
        """ Constructor for save
        
        :param path: path to save folder
        :param should_exist: whether the folder should exist or not
        """
        if should_exist and not path.exists():
            raise FileNotFoundError("Save directory does not exist but should")
        elif not should_exist and path.exists():
            raise FileNotFoundError("Save directory exists but should not")
        
        path.mkdir(parents=True, exist_ok=True)
        self.folder = path
        self.settings_file = path / Path('settings.json')
        self.save_file = path / Path('embeddings.npy')
    
    def write_settings(self, settings):
        """ Writes JSON file for experiment settings """
        with open(self.settings_file, 'w') as f:
            json.dump(settings, f, indent=4)
