import pandas as pd
import numpy as np
from tempeh.configurations import datasets


if __name__ == '__main__':
    compas_dataset = datasets["compas"]()
    X_train, X_test = compas_dataset.get_X(format=pd.DataFrame)
    y_train, y_test = compas_dataset.get_y(format=pd.Series)
    (sensitive_features_train,sensitive_features_test) = compas_dataset.get_sensitive_features("race", format=pd.Series)


    from sklearn.linear_model import LogisticRegression

    for weight in [1,1.1]:
        estimator = LogisticRegression(solver="liblinear", class_weight={0:1,1:weight})
        estimator.fit(X_train, y_train)

        y_predicted = pd.DataFrame(estimator.predict(X_test), columns=['predicted label'])
        testData = pd.concat([X_test, y_test, y_predicted, sensitive_features_test], axis=1)
        testData.to_csv('compas_test_data_weight'+str(weight)+'.csv')
