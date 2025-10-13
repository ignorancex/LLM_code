import numpy as np
import pickle
from aeon.transformations.collection.shapelet_based import RandomShapeletTransform
from utils.data_utils import read_UCR_UEA

train_x, test_x, train_y, test_y, enc1 = read_UCR_UEA('GunPoint', None)
test_y = np.argmax(test_y, axis=1)
train_y = np.argmax(train_y, axis=1)
ST_attribution = RandomShapeletTransform(min_shapelet_length=10,
                                         max_shapelet_length=20,
                                         time_limit_in_minutes=5.0,
                                         random_state=42)
X_new = ST_attribution.fit_transform(train_x, train_y)

shapelet_name = 'RSTF'
with open(f'shapelets/GunPoint/{shapelet_name}.pkl', 'wb') as f:
    pickle.dump(ST_attribution, f)

# shapelet information gain,
# shapelet length,
# start position the shapelet was extracted from,
# shapelet dimension,
# index of the instance the shapelet was extracted from in fit,
# class value of the shapelet,
# The z-normalised shapelet array)

with open(f'shapelets/GunPoint/{shapelet_name}.pkl', 'rb') as f:
    ST_attribution = pickle.load(f)

info_gain, shapelet_length, start_pos, dimension, inst_index, shapelet_class, z_norm_shapelet = \
ST_attribution.shapelets[0]
