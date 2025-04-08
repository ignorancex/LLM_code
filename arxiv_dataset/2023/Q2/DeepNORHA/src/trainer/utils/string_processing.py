import re
from os import path



def parse_boolean(input_string: str): 
    """Parse a given string into a boolean variable

    Args:
        input_string (str): any string containing "true", "TRUE", "false", "FALSE"

    Returns:
        bool: boolean indicating if input_string was true or false
    """

    return input_string.upper()=="TRUE"


    
def list_to_string(input_list: list):
    """Turn input_list of strings into a single string with comma separated values

    Args:
        input_list (list): list of string values

    Returns:
        str: a single string where each element of input_list is comma separated
    """

    if (input_list == None) or (len(input_list)==0):
        return ""
    else:
        return ",".join(list(input_list))



def string_to_list(input_string: str):
    """Turn a string with comma separated values into a list

    Args:
        input_string (str): string with comma separated elements

    Returns:
        list: a list with each of the elements that were comma separated in input_string
    """
    
    # separate elements in a list
    list_to_return = input_string.split(",")
    # if the list has a single element and the element is an empty string, return an empty list
    if (len(list_to_return) == 1) and (list_to_return[0]==""):
        list_to_return = []

    return list_to_return



def remove_extensions(filenames: list):
    """Remove file extensions from each string on a list of filenames

    Args:
        filenames (list): list of strings, each representing a filename (with or without path)

    Returns:
        list: the same list of strings but with each filename without extension
    """

    # initialize an empty list of files
    new_filenames = []
    # iterate for each filename 
    for i in range(len(filenames)):
        # get the filename and the extension
        filename_without_extension, file_extension = path.splitext(filenames[i])
        # append only the filename to the list
        new_filenames.append(filename_without_extension)
    
    return new_filenames



def natural_key(string_: str):
    """Key to use for sorting strings using natural ordering. 
    See http://www.codinghorror.com/blog/archives/001018.html

    Args:
        string_ (str): input string

    Returns:
        int or str: key for sorting
    """
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]
