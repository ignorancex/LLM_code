dataset_to_classname_dict = {}
dataset_to_classname_dict["imagenette"] = ["English springer", "French horn", "cassette player", "chain saw", "church", "garbage truck", "gas pump", "golf ball", "parachute", "tench"]
dataset_to_classname_dict["imagewoof"] = ["Shih-Tzu", "Rhodesian ridgeback", "Beagle", "English foxhound", "Australian terrier", "Border terrier", "Golden retriever", "Old English sheepdog", "Samoyed", "Dingo"]

def get_classnames(dataset: str):
    return dataset_to_classname_dict[dataset]