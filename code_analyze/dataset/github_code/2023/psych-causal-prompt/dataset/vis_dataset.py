
from datasets import load_from_disk

dataset_tiny = load_from_disk('./yelp_tiny_test')
dataset_tiny_new = load_from_disk('./yelp_tiny_test_new')

print(dataset_tiny[:]['label'])
print(dataset_tiny_new[:]['label'])