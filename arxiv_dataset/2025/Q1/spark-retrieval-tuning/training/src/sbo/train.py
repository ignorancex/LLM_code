import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.metrics import mean_squared_error


data = pd.read_csv("src/sbo/features_train.csv")

target = data['value'].to_numpy()
features = data.drop(columns=['applicationName', 'applicationId', 'value']).to_numpy()

groups = data['applicationName'].to_numpy()

logo = LeaveOneGroupOut()

model = RandomForestRegressor(random_state=0)


param_grid = {
    'max_depth': [10, 20, None], 
    'max_features': [1.0, 'sqrt'], 
    'n_estimators':  [10, 20, 50, 100, 200]
}

grid_search = GridSearchCV(
    estimator=model, param_grid=param_grid, cv=logo, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1
).fit(features, target, groups=groups)

print("Best params: ", grid_search.best_params_)
print("Best Training score:", grid_search.best_score_)

best_model = grid_search.best_estimator_


with open('src/sbo/model.pkl', 'wb') as file:
    pickle.dump(best_model, file)

test_data = pd.read_csv("src/sbo/features_test.csv")
test_target = test_data['value'].to_numpy()
test_features = test_data.drop(columns=['applicationName', 'applicationId', 'value']).to_numpy()

test_prediction = best_model.predict(test_features)

print("Best validation score: ", mean_squared_error(test_target, test_prediction))