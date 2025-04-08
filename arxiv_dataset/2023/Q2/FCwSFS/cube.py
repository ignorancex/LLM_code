from FCwSFS import FCwSFS
from datasets import gen_cube
import numpy as np
import warnings

warnings.filterwarnings('ignore')


X_train, y_train = gen_cube(data_points=5000, sigma=0.3)
X_test, y_test = gen_cube(data_points=100, sigma=0.3)


model = FCwSFS(distance_threshold=0.3)

model.fit(X_train=X_train, y_train=y_train)

for f in range(1,20):
    y_pred, features, result = model.evaluate(X_test, y_test, max_features=f, certainty_threshold=0.4)
    
model.save_results("results/FCwSFS-cube_dataset.json")