import numpy as np
from sklearn.utils import shuffle
from sklearn.datasets import fetch_openml


# function for generating cube dataset

def gen_cube(n_features=20, data_points=20000, sigma=0.1, seed=123):
    assert n_features >= 10, 'cube data have >= 10 num of features'
    np.random.seed(seed)
    clean_points = np.random.binomial(1, 0.5, (data_points, 3))
    labels = np.dot(clean_points, np.array([1,2,4]))
    points = clean_points + np.random.normal(0, sigma, (data_points, 3))
    features = np.random.rand(data_points, n_features)
    for i in range(data_points):
        offset = labels[i]
        for j in range(3):
            features[i, offset + j] = points[i, j]
    return features, labels

def get_syntethic_dataset(N=100, features=[2, 0, 1, 3, 4], percents=[30, 25, 20, 15, 10], num_features= 10):
  X_total = []
  y_total = []
  for index, p in enumerate(percents):
    class_samples_num = int(N*p/100) 
    X_calss = np.random.randint(0, 2, (class_samples_num, num_features))
    X_calss[:,features[:index]] = 0
    X_calss[:,features[index]] = 1
    y_class = np.full(class_samples_num, index)
    X_total.append(X_calss)
    y_total.append(y_class)
  X = np.vstack(X_total)
  y = np.hstack(y_total)
  X_shuffled, y_shuffled = shuffle(X, y, random_state=42)

  return X_shuffled, y_shuffled

def get_mnist():
   mnist = fetch_openml('mnist_784', version=1)
   return mnist.data, mnist.target


if __name__ == "__main__":
    data = gen_cube()
    print(data)
