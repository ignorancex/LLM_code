# -*- coding: utf-8 -*-

###############################################################################
# Imports
###############################################################################
import json


###############################################################################
# Functions
###############################################################################

"""
This function allows the user to save one tag dicom in an archive json
"""


def add_tags_dicom(tag_dict, json_file_path):
    json_file = load_json(json_file_path)
    for tag, value in tag_dict.items():
        json_file[tag] = value
    save_json(json_file, json_file_path)


"""
This function allow the user to obtain one tag dicom from an archive json


Return a dict of the tag dicom
"""

def get_tag_dicom(dict_json, json_file_path):
    return {tag:dict_json.get(tag, {"Value":["n/a"]}) for tag in tags}




"""
This function allow the user store one object in an archive json
"""
def save_json(subject, path):
    string_json = json.dumps(subject, default=lambda o: o.__dict__,
         sort_keys=True)
    try:
        f = open(path, "w")
    except IOError:
        f = open(path, "a")
    f.write(string_json)
    f.close()


"""
This function allow the user obtain one object from an archive json

Return the struct dict json
"""
def load_json(path):
    with path.open('r') as file_:
        json_ = json.load(file_)
    return json_
