from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import time
import json

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    

class FCwSFS:
    def __init__(self, distance_threshold = 0.5):
        self.selector = SelectKBest(score_func=f_classif, k=1)
        self.distance_threshold = distance_threshold
        self.results = {
            "train": [],
            "test": []
        }
    
    # it is lazy approach so we only normalize the training data durring training phase
    def fit(self, X_train, y_train): 
        start = time.time()
        self.scaler = MinMaxScaler()
        self.X_train = self.scaler.fit_transform(X_train)
        self.y_train = y_train
        self.num_classes = len(set(y_train))
        end = time.time()
        # print(self.X_train)

        self.results["train"].append({"time" : end - start, "num_samples": len(X_train), "num_features": X_train.shape[1], "num_classes": self.num_classes})
    
    def evaluate(self, X_test, y_test, max_features = 5, certainty_threshold = 1):
        y_predicted = []
        all_selected_features = []
        X_test_scaled = self.scaler.transform(X_test)
        run_times = []
        for x_t in tqdm(X_test_scaled):
            start = time.time()
            selected_faetures = []
            predicted_y = []
            X_train_sub = self.X_train.copy()
            y_train_sub = self.y_train.copy()
            for f in range(max_features):
                if(len(y_train_sub) == 0):
                    break
                X_train_sub[:, selected_faetures] = 0 # this is because the selected feature would not be selected in next phase
                self.selector.fit_transform(X_train_sub, y_train_sub)
                next_feature = self.selector.get_support(indices=True)[0]
                selected_faetures.append(next_feature)
                
                diff = X_train_sub[:, selected_faetures[-1:]] - x_t[selected_faetures[-1:]]
                distances = np.linalg.norm(diff, axis = 1)
                selected_samples = distances < self.distance_threshold
                X_train_sub = X_train_sub[selected_samples, :]
                y_train_sub = y_train_sub[selected_samples]
                probs = np.bincount(y_train_sub, minlength=self.num_classes) / len(y_train_sub)
                predicted_y.append(np.argmax(probs))
                if(np.max(probs)  > certainty_threshold):
                    break
                # print(y_train_sub)
            # print("selected_features", selected_faetures)
            all_selected_features.append(selected_faetures)
            y_predicted.append(predicted_y[-1])
            end = time.time()
            run_times.append(end-start)
        accuracy = np.sum(y_predicted == y_test) / len(y_test)
        self.results["test"].append({
            "max_features" : max_features,
            "certainty_threshold": certainty_threshold,
            "num_samples": len(y_test),
            "num_all_features": X_test.shape[1],
            "avg_selected_features": np.mean([len(s) for s in all_selected_features]),
            "average_test_for_each_sample": np.mean(run_times),
            "accuracy": accuracy,
            "y_test_sample": [y for y in y_test[:5]],
            "y_predicted_sample": [y for y in y_predicted[:5]],
        })
        return np.array(y_predicted), all_selected_features, self.results
    
    def save_results(self, model_path_name ):
        with open(model_path_name, 'w') as f:
            json.dump(self.results, f, cls=NpEncoder)


