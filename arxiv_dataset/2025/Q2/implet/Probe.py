
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import pickle
import utils


class ShapeletProber:
    def __init__(self, model, device='cuda', save_path='probe/simu_implet_random', is_verbose=False):
        self.model = model
        self.device = device
        self.save_path = save_path
        self.is_verbose = is_verbose

        # Ensure save path exists
        if save_path and not os.path.isdir(save_path):
            os.makedirs(save_path, exist_ok=True)

    def compute_distances(self, data, shapelet, length, position):
        distances = np.zeros(data.shape[0])
        for i in range(data.shape[0]):
            distances[i], _ = utils.compute_shapelet_distance(data[i], shapelet, length=length, position=position)
        return distances

    def determine_threshold(self, data, use_info_gain, best_threshold):
        if use_info_gain:
            return best_threshold
        kmeans = KMeans(n_clusters=2, random_state=0).fit(data.reshape(-1, 1))
        return np.mean(kmeans.cluster_centers_.flatten())

    def get_latent_representation(self, dataset, pdata):
        dataset_latent = utils.get_hidden_layers(model=self.model, hook_block=None, data=dataset, device=self.device)
        pdata_latent = utils.get_hidden_layers(model=self.model, hook_block=None, data=pdata, device=self.device)
        return dataset_latent.reshape(dataset_latent.shape[0], -1), pdata_latent.reshape(pdata_latent.shape[0], -1)

    def train_classifier(self, X_train, y_train, classifier_type):
        if classifier_type == 'LR':
            classifier = LogisticRegression()
        elif classifier_type == 'SVC':
            classifier = SVC(kernel='rbf')
        else:
            raise ValueError("Unspecified probing classifier type")
        classifier.fit(X_train, y_train)
        return classifier

    def probe(self, dataset, labels, pdata, shapelet=None, pos=None, shapelet_labels=None, is_threshold_info_gain=False,
              classifier_type='LR', save_results=True, use_saved_latent=False, merge_latent=False):


        # Initialize variables to None
        dataset_s_distances = pdata_s_distances = dataset_s_label = None
        dataset_latent = pdata_latent = None
        X_train = X_test = y_train = y_test = None

        # Check if saved latent data should be used
        latent_data_path = os.path.join(self.save_path, 'probe_latent_label.pkl')
        if use_saved_latent and os.path.exists(latent_data_path):
            if self.is_verbose:
                print("Using saved latent data from:", latent_data_path)
            saved_data = self.load_pickle(latent_data_path)

            # Assign variables from saved data
            dataset_s_label = saved_data['dataset_s_label']
            dataset_latent = saved_data['dataset_latent']
            X_train = saved_data['latent_train']
            X_test = saved_data['latent_test']
            y_train = saved_data['label_train']
            y_test = saved_data['label_test']
        else:
            # Compute distances
            if shapelet_labels is None:
                length = len(shapelet)
                pdata_s_distances = self.compute_distances(pdata, shapelet, length, pos)
                dataset_s_distances, _, best_threshold = utils.get_distances_info_gain(
                    dataset, shapelet, length, pos, labels)

                # Determine threshold
                threshold = self.determine_threshold(pdata_s_distances, is_threshold_info_gain, best_threshold)

                # Label shapelets
                dataset_s_label = [
                    i >= threshold for i in dataset_s_distances]
                pdata_s_label =[
                    i >= threshold for i in pdata_s_distances]
            else:
                dataset_s_label = shapelet_labels[0]
                pdata_s_label = shapelet_labels[1]
            # Extract latent spaces
            dataset_latent, pdata_latent = self.get_latent_representation(dataset, pdata)

            if merge_latent:
                # Merge dataset_latent and pdata_latent
                pdata_latent = np.vstack([dataset_latent, pdata_latent])
                pdata_s_label = np.concatenate([dataset_s_label, pdata_s_label])

            # Prepare data for probing classifier
            X_train, X_test, y_train, y_test = train_test_split(
                pdata_latent,
                pdata_s_label,
                test_size=0.2,
                random_state=42
            )

            # Save latent data if required
            if save_results:
                probe_latent_label = {
                    'dataset_s_distances': dataset_s_distances,
                    'pdata_s_distances': pdata_s_distances,
                    'dataset_s_label': dataset_s_label,
                    'dataset_latent': dataset_latent,
                    'pdata_s_label': pdata_s_label,
                    'pdata_latent': pdata_latent,
                    'latent_train': X_train,
                    'latent_test': X_test,
                    'label_train': y_train,
                    'label_test': y_test,
                }
                self._save_pickle(probe_latent_label, 'probe_latent_label.pkl')

        # Train probing classifier
        classifier = self.train_classifier(X_train, y_train, classifier_type)
        pred_train = classifier.predict(X_train)
        pred_test = classifier.predict(X_test)
        dataset_s_pred = classifier.predict(dataset_latent)

        # Compute accuracies
        train_acc = accuracy_score(y_train, pred_train)
        test_acc = accuracy_score(y_test, pred_test)
        p_data_acc = accuracy_score(dataset_s_label, dataset_s_pred)

        if self.is_verbose:
            print(f"Training Accuracy = {train_acc:.2f}")
            print(f"Testing Accuracy = {test_acc:.2f}")
            print(f"----------------------")
            print(f"Original data Accuracy = {p_data_acc:.2f}")

        results = {
            'pred_train': pred_train,
            'pred_test': pred_test,
            'dataset_s_pred': dataset_s_pred,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'accuracy': p_data_acc,
            'classifier': classifier
        }

        # Optionally save results
        if save_results:
            self._save_pickle(results, 'results.pkl')

        return results

    def _save_pickle(self, data, filename):
        filepath = os.path.join(self.save_path, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load_pickle(self, filename):
        filepath = os.path.join(self.save_path, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The file {filepath} does not exist.")
        with open(filepath, 'rb') as f:
            return pickle.load(f)



def probe_shapelet(dataset, labels, pdata, model, shapelet, pos=None, device='cuda',
                   save_path='probe/simu_implet_random',
                   shapelet_labels=None, is_threshold_info_gain=False, classifier_type='LR', is_verbose=False):
    """
    Probing classifier of the shapelet.

    :param dataset: original dataset
    :param labels: original task label
    :param pdata: simulated diverse datasets that used for probing
    :param model: model to be tested
    :param shapelet: the shapelet that we are investigating
    :param pos: starting position of the shapelet (used for the quick shapelet distance computation)
    :param device:
    :param save_path:
    :param shapelet_labels: if the datasets and pdata is given, if not, computing shapelet distance
     and set up threshold to assign labels
    :param is_threshold_info_gain: if use information gain to set up threshold, if not use k-means.
    :return: return the dictionary that with classification results, prediction and classifiers.
    """
    path_only = save_path
    if path_only and not os.path.isdir(path_only):
        os.makedirs(path_only, exist_ok=True)

    length = len(shapelet)
    num = pdata.shape[0]
    pdata_s_distances = np.zeros(num)
    for i in range(num):
        pdata_s_distances[i], _ = utils.compute_shapelet_distance(pdata[i], shapelet, length=length, position=pos)

    dataset_s_distances, _, best_threshold = utils.get_distances_info_gain(dataset, shapelet, length, pos, labels)

    data = pdata_s_distances.reshape(-1, 1)

    if not is_threshold_info_gain:
        kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
        centroids = kmeans.cluster_centers_.flatten()
        threshold = np.mean(centroids)
    else:
        threshold = best_threshold

    dataset_s_label = shapelet_labels[0] if shapelet_labels[0] is not None else [i >= threshold for i in
                                                                                 dataset_s_distances]
    pdata_s_label = shapelet_labels[1] if shapelet_labels[1] is not None else [i >= threshold for i in
                                                                               pdata_s_distances]

    # extrct latent space
    print(pdata_s_label)
    dataset_latent = utils.get_hidden_layers(model=model, hook_block=None, data=dataset, device=device)
    pdata_latent = utils.get_hidden_layers(model=model, hook_block=None, data=pdata, device=device)

    dataset_latent = dataset_latent.reshape(dataset_latent.shape[0], -1)
    pdata_latent = pdata_latent.reshape(pdata_latent.shape[0], -1)
    print(dataset_latent.shape, pdata_latent.shape)

    # set up for the probing classifier ---> Dataset
    X_train, X_test, y_train, y_test = train_test_split(
        pdata_latent, pdata_s_label, test_size=0.2, random_state=42)

    probe_latent_label = {
        'dataset_s_distances': dataset_s_distances,
        'pdata_s_distances': pdata_s_distances,
        'dataset_s_label': dataset_s_label,
        'dataset_latent': dataset_latent,
        'pdata_s_label': pdata_s_label,
        'pdata_latent': pdata_latent,
        'latent_train': X_train,
        'latent_test': X_test,
        'label_train': y_train,
        'label_test': y_test,
    }
    with open(os.path.join(save_path, 'probe_latent_label.pkl'), 'wb') as f:
        pickle.dump(probe_latent_label, f)

    # set up for the probing classifier ---> Classifier
    if classifier_type == 'LR':
        classifier = LogisticRegression()
    elif classifier_type == 'SVC':
        classifier = SVC(kernel='rbf')
    else:
        raise ValueError('Unspecified probing classifier')
    classifier.fit(X_train, y_train)
    pred_train = classifier.predict(X_train)
    pred_test = classifier.predict(X_test)

    dataset_s_pred = classifier.predict(dataset_latent)
    train_acc = accuracy_score(y_train, pred_train)
    test_acc = accuracy_score(y_test, pred_test)
    p_data_acc = accuracy_score(dataset_s_label, dataset_s_pred)

    if is_verbose:
        print(f'Training Accuracy = {train_acc:.2f}')
        print(f'Testing Accuracy = {test_acc:.2f}')
        print(f"----------------------")
        print(f"original data Accuracy = {p_data_acc:.2f}")

    results = {
        'pred_train': pred_train,
        'pred_test': pred_test,
        'dataset_s_pred': dataset_s_pred,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'accuracy': p_data_acc,
        'classifier': classifier
    }

    with open(os.path.join(save_path, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    return results
