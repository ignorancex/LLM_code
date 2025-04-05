import pyximport

pyximport.install()

from remove_labels import remove_labels
import time
from raise_utils.transforms import Transform
from util import remove_labels_legacy, defect_file_dic, load_defect_data


data = load_defect_data('ivy')
transform = Transform('wfo')
transform.apply(data)
transform.apply(data)
transform = Transform('smote')
transform.apply(data)

a = time.time()
remove_labels(data.x_train, data.y_train)
b = time.time()

print('New:', b - a)

a = time.time()
remove_labels_legacy(data)
b = time.time()

print('Old:', b - a)

