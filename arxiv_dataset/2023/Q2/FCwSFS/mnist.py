from FCwSFS import FCwSFS
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

mnist = fetch_openml('mnist_784', version=1)

# split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(mnist.data, np.array([int(t) for t in mnist.target]), test_size=0.125, random_state=42)

# print(y_train)

model = FCwSFS(distance_threshold=.2)

model.fit(X_train=X_train, y_train=y_train)

accuracies = []
for i in np.arange(1, 100, 10):
    model.evaluate(X_test,y_test, max_features=i, certainty_threshold=1)


model.save_results("results/FCwSFS-mnist.json")
