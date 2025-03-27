
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.metrics import f1_score, recall_score
from keras.utils import to_categorical




x_train, y_train, x_dev, x_test,y_test = pickle.load(open('./pickle/stacking_local.pickle', 'rb'))


def accuracy(original, predicted):
    print("F1_macro score is: " + str(f1_score(original, predicted, average='macro')))
    print("F1_weighted score is: " + str(f1_score(original, predicted, average='weighted')))
    print("recall score is: " + str(recall_score(original, predicted, average='macro')))


def trainandTest(X_train, y_train, X_test):
    # 'learning_rate': 0.011, 'n_estimators': 10, 'max_depth': 4, 'min_child_weight': 1, 'seed': 0,
    # 'subsample': 0.7, 'colsample_bytree': 0.7, 'gamma': 1.7, 'reg_alpha': 1e-05, 'reg_lambda': 1,
    # 'scale_pos_weight': 1
    # XGBoost训练过程，下面的参数就是刚才调试出来的最佳参数组合i

        model = xgb.XGBClassifier(learning_rate= 0.02, n_estimators= 16, max_depth=3, imin_child_weight= 0, seed= 14,
                    subsample= 0.029, colsample_bytree=0.469, gamma=0.4, reg_alpha=0.015, reg_lambda=0.031,
                    scale_pos_weight= 1)
        model.fit(X_train, y_train)

            # 对测试集进行预
        ans = model.predict(X_test)

        accuracy(ans,y_test )



if __name__ == '__main__':
    trainandTest(x_train, y_train.argmax(axis=1),x_test)



