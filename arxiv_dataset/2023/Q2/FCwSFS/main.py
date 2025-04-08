from FCwSFS import FCwSFS
from datasets import get_syntethic_dataset
import numpy as np
import warnings

warnings.filterwarnings('ignore')


X_train, y_train = get_syntethic_dataset(5000)
X_test, y_test = get_syntethic_dataset(100)


model = FCwSFS(distance_threshold=0.3)

model.fit(X_train=X_train, y_train=y_train)

for f in range(1,10):
    y_pred, features, result = model.evaluate(X_test, y_test, max_features=f, certainty_threshold=0.8)
    
model.save_results("results/FCwSFS-syntethic_dataset.json")