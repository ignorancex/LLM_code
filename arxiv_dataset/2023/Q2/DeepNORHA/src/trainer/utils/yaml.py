import os
import yaml

from torch import Tensor

from .string_processing import list_to_string


def load(file_path):
    calc_path = lambda path: os.path.join(os.path.dirname(file_path), path)

    def loadTensor(loader, node):
        return Tensor(loader.construct_sequence(node, deep=True))

    def loadInclude(loader, node):
        path = loader.construct_python_str(node)
        path = calc_path(path)
        return load(path)
    
    class Loader(yaml.FullLoader):
        pass
    
    Loader.add_constructor(u'!Tensor', loadTensor)
    Loader.add_constructor(u'!include', loadInclude)
    
    with open(file_path, 'r') as f:
        return yaml.load(f, Loader=Loader)



def save_split_file_in_yaml(split_filename : str, dataset_path : str, training_filenames : list, validation_filenames : list, test_filenames : list, verbose : bool = False):
    """Generates a YAML file with a training, validation and test partition

    Args:
        split_filename (str): full path to the file to be saved
        dataset_path (str): path where the files are stored
        training_filenames (list): names of the files for the training set
        validation_filenames (list): names of the files for the validation set
        test_filenames (list): names of the files for the test set
        verbose (bool, optional): use this parameter to show the number of images in train, val and test. Defaults to False.

    Returns:
        dict: dictionary with all the content of the YAML file
    """

    # make a dictionary with the content for the file
    split_file_content = {"path": dataset_path,
                          "training" : list_to_string(training_filenames),
                          "validation" : list_to_string(validation_filenames),
                          "test" : list_to_string(test_filenames)}

    # export the split
    with open(split_filename, "w") as file:
        yaml.dump(split_file_content, file)

    # print the split
    if verbose:
        print("- Training set: {} files".format(len(training_filenames)))
        print("- Validation set: {} files".format(len(validation_filenames)))
        print("- Test set: {} files".format(len(test_filenames)))

    return split_file_content
