# public packages
import json, sys, os
import numpy as np
# self-defined packages
sys.path.append(os.path.abspath('code'))
import helper_funcs, glass
import data.bnci2014008 as data

for subject_id in range(1, 9):
    print(f'Fitting GLASS for subject {subject_id}')
    # process data
    X, y = data.get_processed_data(subject_id, sfreq=128, fmin=0.1, fmax=24)
    X = X[:, :, :, :, :, 39:]
    X*=1e6
    n_channel, n_time = X.shape[-2:]
    X_train, y_train, X_test, y_test = X[:7], y[:7], X[7:], y[7:]
    # fit model without shrinkage to determine cutoff
    model = glass.Glass()
    model.process_data(X_train.reshape([-1, 6, n_channel, n_time]), y_train.reshape([-1, 6]))
    model.mfvb(2000, learning_rate=0.05, seed=1)
    cutoff = np.median(np.abs(model.betaMat))
    # fit model
    model = glass.Glass(shrinkage_factor=0.5*cutoff)
    model.process_data(X_train.reshape([-1, 6, n_channel, n_time]), y_train.reshape([-1, 6]))
    model.mfvb(2000, learning_rate=0.05, seed=1)
    # evaluate result
    yhat_test = model.predict_prob(X_test.reshape([-1, 6, n_channel, n_time]))
    yhat_test = yhat_test.reshape(y_test.shape)
    result = helper_funcs.evaluate(yhat_test, y_test)
    # save result
    with open(f'output/subject_{subject_id}.json', 'w') as f:
        json.dump(result, f)