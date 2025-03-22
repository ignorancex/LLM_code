""" This class parses the json annotation file.
'id', 'container id', 'width at the top', 'width at the bottom', 'height', 'container capacity', 'container mass' are stored in lists.
"""
import json


class JsonParser:
    def __init__(self):
        """ Initializes class fields.
        """
        self.image_name = []
        self.container_id = []
        self.ar_w = []
        self.ar_h = []
        self.avg_d = []
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
            self.image_name.append(d['image_name'])
            self.container_id.append(d['container id'])
            self.ar_w.append(d['aspect ratio width'])
            self.ar_h.append(d['aspect ratio height'])
            self.avg_d.append(d['average distance'])
            self.wt.append(d['width top'])
            self.wb.append(d['width bottom'])
            self.height.append(d['height'])
            self.capacity.append(d['capacity'])
            self.mass.append(d['mass'])
        # Close file
        f.close()
