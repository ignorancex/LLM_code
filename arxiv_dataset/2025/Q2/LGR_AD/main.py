import json
from ODAlgorithms import Algorithm
import diffusers # select this instead of ODAlgorithms for LGR-AD  
from sklearn.metrics import classification_report
import pandas as pd
from datasetloader import DatasetLoader  # Make sure dataset_loader.py is in the same directory or in the PYTHONPATH
import networkx as nx
import numpy as np
from GraphRepresentation import GraphRepresentation
from GCN import *
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import KLDivergence
import warnings
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning

# Ignore warnings of type UndefinedMetricWarning and ConvergenceWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
# Set of algorithms can be defined as a dictionnary/json file
# Define a function to evaluate anomaly detection models
def evaluate_model(model, X_train, X_test, y_test):
    model.fit(X_train)
    y_pred = model.predict(X_test)
    # Reshape the predictions to be in [0, 1] (fraud)
    y_pred = [1 if x == -1 else 0 for x in y_pred]
    print(classification_report(y_test, y_pred))
# Splitting features and labels

for d in ['Titanic-Dataset.csv','heart.csv','Employee.csv','cc_approvals.csv','water_potability.csv']:
    
    # for LGR-AD, use diffusers 
    algorithms = {
        'RandomForestClassifier': {
            'n_estimators': 10,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        },
        'GradientBoostingClassifier': {
            'n_estimators': 10,
            'learning_rate': 0.1,
            'max_depth': 10,
            'random_state': 42
        },
        'LogisticRegression': {
            'penalty': 'l2',
            'C': 1.0,
            'solver': 'lbfgs',
            'max_iter': 100
        },
        'SVC': {
            'C': 1.0,
            'kernel': 'rbf',
            'degree': 3,
            'gamma': 'scale',
            'max_iter': 100  # Note: Setting max_iter here might lead to a warning about not converging.
        },
        'KNeighborsClassifier': {
            'n_neighbors': 5,
            'weights': 'uniform',
            'algorithm': 'auto'
        },
        'DecisionTreeClassifier': {
            'criterion': 'gini',
            'splitter': 'best',
            'max_depth': None,
            'min_samples_split': 2
        },
        'GaussianNB': {
            'var_smoothing': 1e-09
        },
        'MLPClassifier': {
            'hidden_layer_sizes': (100,),
            'activation': 'relu',
            'solver': 'adam',
            'max_iter': 100
        }
    }
    loader = DatasetLoader(file_path=d)
    loader.load_data()
    # This class Create the algorithms and builds them
    algorithms = Algorithm(algorithms,loader)
    algorithms.Train(1,model_type='classification')
    evaluation = algorithms.Evaluate()   
    gr = GraphRepresentation(algorithms,X=algorithms.getFeatures())
    
    
    from networkx.linalg.laplacianmatrix import laplacian_matrix
    from scipy.sparse import csr_matrix
    import matplotlib.pyplot as plt
    
    # Define the custom loss function
    lambda_value = 5.0  # You would set this hyperparameter
    gamma_value = 5.0  # You would set this hyperparameter
        
    
    
    
    def composite_loss(Laplacian):
        def loss(y_true, y_pred):
            # Task-specific loss (e.g., cross-entropy loss)
            task_loss = K.categorical_crossentropy(y_true, y_pred)
            
            # Constraint loss (e.g., Kullback-Leibler divergence)
            constraint_loss = KLDivergence()(y_true, y_pred)
            
            # Laplacian loss
            laplacian_loss = 1/K.sum(K.square(K.cast(Laplacian, 'float32')))
            
            # Total loss
            total_loss = task_loss +( lambda_value * constraint_loss) + (gamma_value * laplacian_loss)
            
            return total_loss
        return loss
    nodes=list(gr.getNodes())
    dfs=[]
    graphs = []
    L=[]
    
    for g in gr.graphs1:
        
        #dfs.append(gr.getEncoding(500, g))
    
        A = nx.adjacency_matrix(g).todense()
        X = np.eye(g.number_of_nodes()) 
        graphs.append((A, X))
        T = nx.maximum_spanning_tree(g)
    
        # Compute the adjacency matrix of the spanning tree
        adj_matrix = nx.adjacency_matrix(T)
        
        # Compute the degree matrix of the spanning tree
        degree_matrix = np.diag([T.degree(n, weight='weight') for n in T.nodes()])
        
        # Compute the Laplacian matrix of the spanning tree
        laplacian_matrix = csr_matrix(degree_matrix - adj_matrix)
        #l = laplacian_matrix(g).toarray()
        L.append(laplacian_matrix.toarray())
    
    from sklearn.model_selection import KFold
    import numpy as np
    def f1_score(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())
        f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
        return f1_val
    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision_value = true_positives / (predicted_positives + K.epsilon())
        return precision_value
    
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall_value = true_positives / (possible_positives + K.epsilon())
        return recall_value
    # Number of splits
    n_splits = 5
    kf = KFold(n_splits=n_splits)
    
    
    adjacency_matrices = np.array([g[0] for g in graphs])
    node_features_list = np.array([g[1] for g in graphs])
    
    adjacency_matrices = adjacency_matrices.astype('float64')
    node_features_list = node_features_list.astype('float64')
    unique_classes = np.unique(algorithms.y_train)
    num_unique_classes = len(unique_classes)
    y = np.array(algorithms.y_train)
    y_one_hot = to_categorical(y, num_classes=num_unique_classes)
    accumulated_metrics = {model_name: {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0} for model_name in algorithms.getAlgorithms()}
    accumulated_metrics['GCN'] = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0}
    all_train_losses = []
    all_val_losses = []
    for train_index, val_index in kf.split(adjacency_matrices):
        train_adjacency, val_adjacency = adjacency_matrices[train_index], adjacency_matrices[val_index]
        train_features, val_features = node_features_list[train_index], node_features_list[val_index]
        train_y, val_y = y_one_hot[train_index], y_one_hot[val_index]
        train_L, val_L = np.array(L, dtype='float32')[train_index], np.array(L, dtype='float32')[val_index]
        
        model = GCN(num_unique_classes)
        model.compile(optimizer='adam', loss=composite_loss(train_L), metrics=['accuracy', precision, recall, f1_score])
    
        history = model.fit(x=[train_adjacency, train_features], y=train_y, epochs=50, batch_size=128, validation_data=([val_adjacency, val_features], val_y),verbose=0)
        all_train_losses.append(history.history['loss'])
        all_val_losses.append(history.history['val_loss'])
       # Accumulate metrics for GCN model
        accumulated_metrics['GCN']['accuracy'] += history.history['val_accuracy'][-1]
        accumulated_metrics['GCN']['precision'] += history.history['val_precision'][-1]
        accumulated_metrics['GCN']['recall'] += history.history['val_recall'][-1]
        accumulated_metrics['GCN']['f1_score'] += history.history['val_f1_score'][-1]
    
        tx, ty = algorithms.X_train.values[train_index], algorithms.y_train[train_index]
        valx, valy = algorithms.X_train.values[val_index], algorithms.y_train[val_index]
    
        evaluation = algorithms.Evaluate1(tx, ty, valx, valy)
        
        # Update accumulated metrics for individual models
        for model_name, metrics in evaluation.items():
            for metric_name, metric_value in metrics.items():
                accumulated_metrics[model_name][metric_name] += metric_value
    
    # Compute average metrics over all folds
    average_metrics = {model_name: {metric_name: metric_value/n_splits for metric_name, metric_value in metrics.items()} for model_name, metrics in accumulated_metrics.items()}
    avg_train_loss = np.mean(all_train_losses, axis=0)
    avg_val_loss = np.mean(all_val_losses, axis=0)
    pd.DataFrame(avg_train_loss).to_csv(f'{d}_train_losses.csv')
    pd.DataFrame(avg_val_loss).to_csv(f'{d}_val_losses.csv')


    # Save to file
    with open(f'{d}_average_metrics.json', 'w') as f:
        json.dump(average_metrics, f)
    # Plotting the average loss
    plt.figure(figsize=(10, 6))
    plt.plot(avg_train_loss, label='Average Training Loss')
    plt.plot(avg_val_loss, label='Average Validation Loss')
    plt.title(f'Average GCN Training and Validation Loss for {d.split(".")[0]} Dataset')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)  # Adding grid
    plt.savefig(f'{d.split(".")[0]}_loss_plot.png', dpi=300, bbox_inches='tight')
    # Print average metrics
    print("Average Metrics Over All Folds:")
    for model_name, metrics in average_metrics.items():
        metrics_str = ', '.join([f"{metric_name}: {metric_value:.4f}" for metric_name, metric_value in metrics.items()])
        print(f"Model: {model_name} | {metrics_str}")
