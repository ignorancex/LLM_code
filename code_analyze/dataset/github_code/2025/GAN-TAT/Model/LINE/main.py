import igraph as ig
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc, precision_score, recall_score
from embed import pipeline
import itertools
from sklearn.model_selection import KFold
import numpy as np




graph_ = r"Data/Network/less_Tclin_Signalink_PIN_graph.graphml"
graph = ig.Graph.Load(graph_, format='graphml')

data = []
for vertex in graph.vs:
    # Get node attributes excluding 'label'
    node_data = {key: vertex[key] for key in vertex.attributes() if key not in ['label', 'id']}
    
    # Ensure 'name' is the first key
    if 'name' in node_data:
        node_data = {'Gene_name': node_data.pop('name'), **node_data}
    
    data.append(node_data)


def aupr_score(y_true, y_pred_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    return auc(recall, precision)

X_ = pd.read_csv(r'Model/LINE/embedding.csv')
# Convert to DataFrame
X = pd.DataFrame(data)


def split_balance(df, n_splits=20, random_state=42, balance_ratio=1):
    np.random.seed(random_state)
    positive_samples = df[df['label'] == 1]
    negative_samples = df[df['label'] == 0]
    
    
    folds = []
    
    for _ in range(n_splits):
        # Select 80% of positive samples
        pos_fold = positive_samples.sample(frac=0.8, random_state=random_state)
        
        # Calculate the number of negative samples needed
        num_neg_samples = int(len(pos_fold) * balance_ratio)
        
        # Randomly select negative samples
        neg_fold = negative_samples.sample(n=num_neg_samples, random_state=random_state)
        
        # Combine positive and negative samples
        fold = pd.concat([pos_fold, neg_fold])
        
        # Shuffle the fold
        fold = fold.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        folds.append(fold)
    
    return folds

def aupr_score(y_true, y_pred_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    return auc(recall, precision)

def compute_metrics(y_true, y_proba, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_proba),
        'aupr': aupr_score(y_true, y_proba)
    }


def custom_cross_validate(X, hype=None, n_splits=5, spliting=20, random_state=42, balance_ratio=1, classifier='XgBoost'):
    # Initialize KFold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    all_metrics = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        
        x_df_train = pd.DataFrame(X_train)
        x_df_test = pd.DataFrame(X_test)

        folds = split_balance(x_df_train, n_splits=spliting, balance_ratio=balance_ratio)
        models = train(folds, hype=hype, classifier=classifier)
        pre_proba = evaluate(x_df_test, models)
        
        y_pred = (pre_proba >= 0.5).astype(int)
        
        metrics = compute_metrics(x_df_test['label'], pre_proba, y_pred)
        all_metrics.append(metrics)

    return all_metrics

def train(folds, hype=None, classifier='XgBoost'):

    models = []

    for i in range(len(folds)):

        X_features = folds[i].drop(['Gene_name'], axis=1)
        y = X_features['label']
        X_features = X_features.drop('label', axis=1)

        if classifier=='XgBoost':
            # Define the XGBoost model
            if hype != None:
                model = xgb.XGBClassifier(**hype)
                model.fit(X_features, y)
            else:
                model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
                model.fit(X_features, y)
                
        elif classifier == 'DT':
            
            if hype is not None:
                model = DecisionTreeClassifier(**hype)
            else:
                model = DecisionTreeClassifier(random_state=42)
            model.fit(X_features, y)

        elif classifier == 'RF':
            
            # Define the Random Forest model
            if hype is not None:
                model = RandomForestClassifier(**hype)
            else:
                model = RandomForestClassifier(random_state=42)
            model.fit(X_features, y)

        models.append(model)

    return models

def evaluate(df_test, models):

    test_features = df_test.drop(['Gene_name'], axis=1)
    y = test_features['label']
    test_features = test_features.drop('label', axis=1)
    predictions_proba = []

    
    for model in models:
        # Predict probabilities
        proba = model.predict_proba(test_features)[:, 1]  # Probabilities for the positive class
        predictions_proba.append(proba)

    predictions_proba = np.array(predictions_proba)
    avg_probas = np.mean(predictions_proba, axis=0)

    return avg_probas

def compute_mean_and_std_metrics(metrics):
    # Initialize dictionaries to hold the sums and squared sums of each metric
    metric_sums = {key: 0 for key in metrics[0].keys()}
    metric_squared_sums = {key: 0 for key in metrics[0].keys()}

    # Sum each metric and squared metric across all folds
    for metric in metrics:
        for key, value in metric.items():
            metric_sums[key] += value
            metric_squared_sums[key] += value ** 2

    n = len(metrics)
    # Calculate the mean for each metric
    metric_means = {key: value / n for key, value in metric_sums.items()}

    # Calculate the standard deviation for each metric
    metric_stds = {
        key: ((metric_squared_sums[key] / n) - (metric_means[key] ** 2)) ** 0.5 for key in metrics[0].keys()
    }

    # Construct the string with mean ± std for each metric
    metric_results = {key: f"{metric_means[key]:.4f} ± {metric_stds[key]:.4f}" for key in metrics[0].keys()}

    return metric_results

def pipeline(X, pos_y, hype=None, cross_split=5, balance_split=30, balance_ratio=1, classifier='XgBoost'):

    positive_labels = pd.read_csv(pos_y, header=None)
    X['label'] = X['Gene_name'].isin(positive_labels[1]).astype(int)
    result = custom_cross_validate(X, hype=hype, n_splits=cross_split, spliting=balance_split, balance_ratio=balance_ratio, classifier=classifier)
    met = compute_mean_and_std_metrics(result)

    for keys, value in met.items():
        print(keys + ': ' + value)

    return met


def grid_search(X, pos_y, param_grid, cross_split=5, balance_split=30, balance_ratio=1, classifier='XgBoost'):
    # Generate all combinations of hyperparameters
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    results = []
    trial = 0
    for params in combinations:
        trial +=1
        # Update the hype dictionary with the current combination of hyperparameters

        '''
        hype = {
            'learning_rate': params['learning_rate'],  # Different learning rates
            'max_depth': params['max_depth'],               # Different depths of trees
            'n_estimators': params['n_estimators'],           # Number of trees
            'subsample': params['subsample'],             # Fraction of samples used for fitting
            'colsample_bytree': params['colsample_bytree'] # Whether bootstrap samples are used when building trees
        }

        hype = {
            'n_estimators': params['n_estimators'],            # Number of trees in the forest
            'max_depth': params['max_depth'],            # Maximum depth of the tree
            'min_samples_split': params['min_samples_split'],           # Minimum number of samples required to split an internal node
            'min_samples_leaf': params['min_samples_leaf'],             # Minimum number of samples required to be at a leaf node
            'max_features': params['max_features'],    # Number of features to consider when looking for the best split
            'bootstrap': params['bootstrap']                 # Whether bootstrap samples are used when building trees
        }'''

        hype = {
            'criterion': params['criterion'],            # Criterion for splitting ('gini' or 'entropy')
            'max_depth': params['max_depth'],     # Maximum depth of the tree
            'min_samples_split': params['min_samples_split'],         # Minimum number of samples required to split an internal node
            'min_samples_leaf': params['min_samples_leaf'],             # Minimum number of samples required to be at a leaf node
            'splitter': params['splitter']
        }



        # Run the pipeline with the current hyperparameters
        met = pipeline(X, pos_y, hype, cross_split=cross_split, balance_split=balance_split, balance_ratio=balance_ratio, classifier=classifier)
        score = float(met['roc_auc'][:-9])

        # Store the result
        print(f"trial: {trial}: score: {score}")
        results.append({'params': params, 'score': score})

    # Sort results by score in descending order
    results = sorted(results, key=lambda x: x['score'], reverse=True)

    best_params = results[0]['params']
    best_score = results[0]['score']

    return best_params, best_score, results


pos_y = r"Data/Labels/Tclin.csv"
pos_y2 = r"Data/Labels/pancreatic intraductal papillary-mucinous neoplasm-Tclin.csv"
pos_y3 = r"Data/Labels/acute myeloid leukemia-tclin.csv"

X__dropped = X_.drop(columns=['node_id'])
df = pd.concat([X, X__dropped], axis=1)
print(1)

Xx = df.iloc[:, 1:-1]  # All columns except the first (name) and the last (label)
y = df.iloc[:, -1]  # Last column (label)

param_grid_xgb = {
    'learning_rate': [0.01, 0.05, 0.1],  # Different learning rates
    'max_depth': [10],               # Different depths of trees
    'n_estimators': [300],           # Number of trees
    'subsample': [0.6, 1.0],             # Fraction of samples used for fitting
    'colsample_bytree': [0.6, 1.0]       # Fraction of features used for fitting
}

param_grid_dt = {
    'criterion': ['entropy'],            # Criterion for splitting ('gini' or 'entropy')
    'max_depth': [3, 7, 10, 12],     # Maximum depth of the tree
    'min_samples_split': [2, 5, 10, 20, 30],         # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4, 8, 12],             # Minimum number of samples required to be at a leaf node
    'splitter': ['best']
}

param_grid_rf = {
    'n_estimators': [300],            # Number of trees in the forest
    'max_depth': [15],            # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],           # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],             # Minimum number of samples required to be at a leaf node
    'max_features': ['sqrt'],    # Number of features to consider when looking for the best split
    'bootstrap': [False]                 # Whether bootstrap samples are used when building trees
}

param, score, log = grid_search(df.iloc[:, :-1], pos_y, param_grid_dt, cross_split=5, balance_split=30, balance_ratio=2, classifier='DT')

print(param, score)