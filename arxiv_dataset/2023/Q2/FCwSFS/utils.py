import numpy as np

# function for geneartion of dataset with partial observation
def get_partial_zeros(X,Y, N=100000, max_selected_features_percent=1, num_selected_features = None):
  num_samples, num_features = X.shape
  partial_x = []
  partial_y = []
  complete_x = []
  for i in range(N):
    num_selected_features = num_selected_features if num_selected_features else int(np.random.rand()*num_features*max_selected_features_percent);
    mask = [int(np.random.rand() < num_selected_features/num_features) for i in range(num_features)]
    random_sample_index = np.random.randint(0, num_samples)
    x = np.copy(X[random_sample_index, :])
    x[np.logical_not(mask)] = 0
    partial_x.append(np.hstack([x, mask]))
    partial_y.append(Y[random_sample_index])
    complete_x.append(X[random_sample_index, :])
  return np.array(partial_x, dtype=np.float32), np.array(partial_y), np.array(complete_x)


if __name__ == '__main__':
    (x_p, y_p, x_c) = get_partial_nan(np.ones((10,3)), np.ones(10), num_selected_features = 2)
    print(x_p.shape)