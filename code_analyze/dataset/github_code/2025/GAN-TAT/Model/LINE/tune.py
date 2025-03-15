import igraph as ig
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc
from Model.LINE.embed import pipeline



def aupr_score(y_true, y_pred_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    return auc(recall, precision)

def tune(graph, negsamplesize=5, dimension=128, batchsize=5, epochs=1, learning_rate=0.025, negativepower=0.75):

    X_ = pipeline(graph, negsamplesize=negsamplesize, dimension=dimension, batchsize=batchsize, epochs=epochs, learning_rate=learning_rate, negativepower=negativepower)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_.iloc[:, 1:-1], X_.iloc[:, -1], test_size=0.2, random_state=42)

    # Create an XGBoost classifier
    xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    xgb_clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = xgb_clf.predict(X_test)
    y_pred_proba = xgb_clf.predict_proba(X_test)[:, 1]

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_ = roc_auc_score(y_test, y_pred_proba)
    aupr = aupr_score(y_test, y_pred_proba)

    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"AUC: {auc_}")
    print(f"AUPR: {aupr}")

    return auc_


