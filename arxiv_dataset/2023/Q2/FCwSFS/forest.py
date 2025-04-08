from sklearn.datasets import fetch_covtype
import numpy as np
from sklearn.model_selection import train_test_split
from FCwSFS import FCwSFS
import warnings
warnings.filterwarnings('ignore')

# Fetch the forest dataset
forest_data = fetch_covtype()

X_train, X_test, y_train, y_test = train_test_split(forest_data.data, np.array([int(t) for t in forest_data.target]), test_size=0.01, random_state=42)


model = FCwSFS(distance_threshold=.3)

model.fit(X_train=X_train[:1000], y_train=y_train[:1000])

accuracies = []
for i in np.arange(1, 30, 2):
    model.evaluate(X_test[:100],y_test[:100], max_features=i, certainty_threshold=1)


model.save_results("results/FCwSFS-forest.json")