# create a class mapping.
import pandas as pd


def create_mapping(path = "class/full_classes.txt"):
	# return list with (id, class) tuples

	with open(path) as f:
		classes = f.readlines()
	classes = [x.strip() for x in classes]
	classes = [x for x in classes if x]
	classes = [(id,x) for id,x in enumerate(classes)]
	return classes


def save_mapping(mapping, path = "class/mapping.txt"):
	# stores mapping in human readable format

	mapping = pd.DataFrame(mapping, columns=["id", "class"])
	mapping = mapping.set_index("id")
	mapping.to_csv(path)


def load_mapping(path="class/mapping.txt"):
	mapping = pd.read_csv(path)
	mapping = mapping.set_index('id').T.to_dict('list')
	mapping = {k:v[0] for k,v in mapping.items()}
	return mapping


def get_class(index):
	return idx2class[index]


def get_index(class_str):
	return class2idx[class_str]

def get_num_classes():
	return len(class2idx.keys())


if __name__ == "__main__":
	save_mapping(create_mapping())

else:	
	# load dictionaries once for faster access.
	idx2class = load_mapping()
	class2idx = {v:k for k,v in idx2class.items()}

