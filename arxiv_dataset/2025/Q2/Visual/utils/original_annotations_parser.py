""" This class parses the original json annotation file, the one provided by organizers.
'id', 'container id', 'width at the top', 'width at the bottom', 'height', 'container capacity', 'container mass' are stored in lists.
"""
import json


class JsonParser:
    def __init__(self):
        """ Initializes class fields.
        """
        self.video_id = []
        self.container_id = []
        self.wt = []  # width top
        self.wb = []  # width bottom
        self.height = []  # height
        self.capacity = []
        self.mass = []

    def load_json(self, path_to_json):
        """ Stores information about the annotation json file.

        :param path_to_json: path to json file
        :return: None
        """
        # Open JSON file
        f = open(path_to_json)
        # JSON object as a dictionary
        data = json.load(f)
        # Iterate through the json and store info
        for d in data['annotations']:
            self.video_id.append(d['id'])
            self.container_id.append(d['container id'])
            self.wt.append(d['width at the top'])
            self.wb.append(d['width at the bottom'])
            self.height.append(d['height'])
            self.capacity.append(d['container capacity'])
            self.mass.append(d['container mass'])
        # Close file
        f.close()
