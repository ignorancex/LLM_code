import os
import random

import numpy
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tsai.models.FCN import FCN
from tsai.models.MLP import MLP
from tsai.models.RNN import RNN

from Probe import ShapeletProber
from implet_main import implet_main
from utils import selected_training, selected_uni, pickle_save_to_file, generate_loader, fit, get_all_preds, \
    pickle_load_from_file, get_pdata, get_attr, plot_multiple_images_with_attribution


class MainExperiment:
    def __init__(self, feature_shapelet,
                 model_dataset_path,
                 attr_save_dir,
                 probe_dir,
                 target_class,
                 sample_filter_class,
                 sample_filter_by_pred_y,
                 seed=42, device='cuda', model_type='FCN', xai_name='DeepLift', train_model=True,
                 instance_length=None,
                 dataset_name=None,
                 verbose=False,
                 ):
        self.verbose = verbose
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.device = device
        self.model_type = model_type
        self.xai_name = xai_name

        self.each_data_repeat_max = 100

        self.model = None
        self.untrain_model = None
        self.diverse_training = selected_training


        self.dataset_name = dataset_name
        self.diverse_probing = selected_uni
        if (self.dataset_name is not None) and (self.dataset_name in self.diverse_probing): # exclude the dataset in probing if it is in ther
            self.diverse_probing.remove(self.dataset_name)

        self.model_dataset_path = model_dataset_path
        self.attr_save_dir = attr_save_dir

        self.target_class = target_class
        self.sample_filter_class = sample_filter_class
        self.sample_filter_by_pred_y = sample_filter_by_pred_y

        self.probe_dir = probe_dir

        self.feature_shapelet = feature_shapelet

        if train_model or not os.path.isfile(os.path.join(self.model_dataset_path, 'data.pkl')):
            self.instance_length = instance_length
            pdata_dict = self.prepare_data(self.feature_shapelet, self.diverse_training, save_dir=None)
            print("training data shape:", pdata_dict['pdata_ws'].shape)
            data_train = np.concatenate((pdata_dict['pdata_ws'], pdata_dict['pdata_wos']), axis=0)
            data_label = np.array([0] * len(pdata_dict['pdata_ws']) + [1] * len(pdata_dict['pdata_wos']))

            self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
                data_train, data_label, test_size=0.2, random_state=42)

            enc1 = OneHotEncoder(sparse_output=False).fit(self.train_y.reshape(-1, 1))
            self.train_y = enc1.transform(self.train_y.reshape(-1, 1))
            self.test_y = enc1.transform(self.test_y.reshape(-1, 1))
            self.data = {
                'train_x': self.train_x,
                'test_x': self.test_x,
                'train_y': self.train_y,
                'test_y': self.test_y
            }
            pickle_save_to_file(self.data, os.path.join(self.model_dataset_path, 'data.pkl'))

            self.model = self.setup_model(input_length=self.train_x.shape[-1], num_classes=2)
            self.untrain_model = self.setup_model(input_length=self.train_x.shape[-1], num_classes=2)

            train_loader, test_loader = generate_loader(self.train_x, self.test_x, self.train_y, self.test_y,
                                                        batch_size_train=128,
                                                        batch_size_test=32)
            fit(self.model, train_loader, device=device, num_epochs=200)
            torch.save(self.model.state_dict(), f'{self.model_dataset_path}/weight.pt')
            self.test_preds, ground_truth = get_all_preds(self.model, test_loader, device=device)
            self.test_y = np.argmax(self.test_y, axis=1)
            self.train_y = np.argmax(self.train_y, axis=1)
            ground_truth = np.argmax(ground_truth, axis=1)
            np.save(f'{self.model_dataset_path}/test_preds.npy', np.array(self.test_preds))
            acc = accuracy_score(ground_truth, self.test_preds)
            print(f'acc:{acc:.3f}')
            a = classification_report(ground_truth, self.test_preds, output_dict=True)
            dataframe = pd.DataFrame.from_dict(a)
            dataframe.to_csv(f'{self.model_dataset_path}/classification_report.csv', index=False)

        else:
            print(f'read data and model from {self.model_dataset_path}')
            self.data = pickle_load_from_file(os.path.join(self.model_dataset_path, 'data.pkl'))

            self.train_x, self.test_x, self.train_y, self.test_y = (
                self.data['train_x'], self.data['test_x'], self.data['train_y'], self.data['test_y'])
            self.test_preds = np.load(os.path.join(self.model_dataset_path, 'test_preds.npy'))
            self.test_y = np.argmax(self.test_y, axis=1)
            self.train_y = np.argmax(self.train_y, axis=1)

            num_classes = len(np.unique(self.train_y))

            self.model = self.setup_model(input_length=self.train_x.shape[-1], num_classes=num_classes)
            self.untrain_model = self.setup_model(input_length=self.train_x.shape[-1], num_classes=num_classes)
            state_dict = torch.load(f'{self.model_dataset_path}/weight.pt', map_location='cuda:1')
            self.model.load_state_dict(state_dict)
            self.instance_length = self.train_x.shape[-1]

        self.untrain_model.eval()
        self.model.eval()

    def prepare_data(self, shapelet, diverse_datasets: list, save_dir=None):
        return get_pdata(
            shapelet=shapelet,
            selected_datasets=diverse_datasets,
            inst_length=self.instance_length,
            num_shapelet=1,
            is_add=False,
            repeat_max=self.each_data_repeat_max,
            is_z_norm=True,
            is_best_insert=True,
            save_dir=save_dir
        )

    def get_attr(self, is_read=True, is_plot=False):
        """
        Generate attributions and visualize them.

        Args:

            attr_save_dir (str): Directory to save the attributions.

        Returns:
            attr_gp (np.ndarray): Generated attributions.
        """

        if not os.path.isfile(self.attr_save_dir) or not is_read:  # True:  #
            attr_gp, _ = get_attr(self.model, self.test_x, None, None,
                                  save_dir=self.attr_save_dir,
                                  xai_name=self.xai_name, target_class=self.target_class)
        else:
            print(f'read attr from {self.attr_save_dir}')
            attr = pickle_load_from_file(self.attr_save_dir)
            attr_gp = attr['attributions']

        # Visualization
        if is_plot:
            plot_indices = np.arange(8)
            plot_multiple_images_with_attribution(
                self.test_x[plot_indices],
                self.test_y[plot_indices],
                8,
                figsize=(12, 6),
                use_attribution=True,
                attributions=attr_gp[plot_indices],
                normalize_attribution=True,
                save_path=None,
                test_y=self.test_preds[plot_indices],
                startings=[[30]] * 8,
                shapelet_length=15
            )

        return attr_gp

    def setup_model(self, input_length, num_classes):
        if self.model_type == 'RNN':
            return RNN(c_in=1, c_out=num_classes)
        elif self.model_type == 'FCN':
            return FCN(c_in=1, c_out=num_classes)
        elif self.model_type == 'MLP':
            return MLP(c_in=1, c_out=num_classes, seq_len=input_length)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def probe_features(self, model, test_x, test_y, pdata, save_dir, shapelet_labels):
        prober = ShapeletProber(model, device=self.device, save_path=save_dir, is_verbose=False)
        results = prober.probe(
            dataset=test_x,
            labels=test_y,
            pdata=pdata,
            shapelet=None,
            pos=None,
            shapelet_labels=shapelet_labels,
            is_threshold_info_gain=False,
            classifier_type='LR',
            save_results=True,
            use_saved_latent=False,
            merge_latent=False
        )
        print(f"Probing Results: Train Accuracy: {results['train_acc']}, Test Accuracy: {results['test_acc']}")
        return results

    def get_implets(self, implet_names, is_read_implet=False, lamb=0.1, thresh_factor=1.0):

        if is_read_implet and os.path.isfile(os.path.join(self.probe_dir, 'implets_cluster_results.pkl')):
            # implet clusters
                implet_cluster_results = pickle_load_from_file(os.path.join(self.probe_dir, 'implets_cluster_results.pkl'))
                # implets
                implets = pickle_load_from_file(os.path.join(self.probe_dir, 'implets.pkl'))['implets']

        else:
            if is_read_implet:
                print('implets not extracted yet')
            implets, implet_cluster_results = implet_main(
                                                          dataset=implet_names, model_path=self.model_dataset_path,
                                                          k=None,
                                                          lamb=lamb, thresh_factor=thresh_factor, model_type='FCN',
                                                          xai_name=self.xai_name,
                                                          attr_class=self.target_class,
                                                          sample_filter_class=self.sample_filter_class,
                                                          sample_filter_by_pred_y=True,
                                                          attr_save_dir=self.attr_save_dir,
                                                          implets_save_dir=self.probe_dir,
                                                          verbose=self.verbose,
                                                          is_plot=True
                                                           )
            implets = implets['implets']
        return implets, implet_cluster_results

    def probe_shapelet(self, shapelet, save_dir, is_read_pdata=False,
                       shapelet_labels_ori=None, is_model_untrained=False):
        # feature probe:
        save_dir_pdata = os.path.join(save_dir, 'pdata.pkl')
        if is_read_pdata and os.path.isfile(save_dir_pdata):
            pdata_dict = pickle_load_from_file(save_dir_pdata)
        else:
            pdata_dict = get_pdata(shapelet=shapelet, selected_datasets=self.diverse_probing,
                                   inst_length=self.instance_length,
                                   num_shapelet=1, is_add=False, repeat_max=100,
                                   is_best_insert=True,
                                   is_z_norm=True, save_dir=save_dir_pdata)
        pdata = np.concatenate((pdata_dict['pdata_ws'], pdata_dict['pdata_wos']), axis=0)

        p_data_labels = np.array([0] * len(pdata_dict['pdata_ws']) + [1] * len(pdata_dict['pdata_wos']))
        if shapelet_labels_ori is None:
            shapelet_labels_ori = self.test_y
        shapelet_labels = [shapelet_labels_ori, p_data_labels]
        # pc
        input_model = self.model if not is_model_untrained else self.untrain_model
        prober = ShapeletProber(input_model, device='cuda', save_path=save_dir, is_verbose=False)
        results = prober.probe(dataset=self.test_x, labels=self.test_y, pdata=pdata, shapelet=None, pos=None,
                               shapelet_labels=shapelet_labels,
                               is_threshold_info_gain=False,
                               classifier_type='LR', save_results=True, use_saved_latent=False, merge_latent=False)

        probe_coef = results['classifier'].coef_.flatten()
        train_acc = results['train_acc']
        test_acc = results['test_acc']
        original_acc = results['accuracy']
        tcav_score = self.TCAV_score(_probe_coef=probe_coef, _model=input_model)
        latent_data_path = os.path.join(save_dir, 'probe_latent_label.pkl')
        data_latent = pickle_load_from_file(latent_data_path)['dataset_latent']
        cf_score = self.Latent_CF_score(_data_latent=data_latent, _probe_classifier=results['classifier'],
                                        _model=input_model)
        if self.verbose:
            print(f'probing training acc: {train_acc:.2f}')
            print(f'probing testing acc: {test_acc:.2f}')
            print(f'original instance acc: {original_acc:.2f}')
            print(f'TCAV score {tcav_score:.2f}')
            print(f'cf_score score {cf_score:.2f}')
            print('-----------------------------------------')
        return results, np.abs(tcav_score), cf_score

    def probe_clusters(self, implet_names, save_dir, is_read_pdata=False, is_read_implet=False, lamb=0.1,
                       thresh_factor=1.0, is_model_untrained=False):

        implets, implet_cluster_results = self.get_implets(implet_names, is_read_implet=is_read_implet,
                                                           lamb=lamb, thresh_factor=thresh_factor)

        cluster_indices = implet_cluster_results['best_indices_dep']
        results = []
        tcav_scores = []
        cf_scores = []
        for i, (cluster_index, cluster_instance) in enumerate(cluster_indices.items()):
            print(f'cluster: {i}')

            clsuter_implets = [implets[i][1].flatten() for i in cluster_instance]
            clsuter_implets_instance_number = [implets[i][0] for i in cluster_instance]
            # for impl in clsuter_implets:
            #     plt.plot(impl,color='blue')
            # plt.show()
            test_x_shapelet_label = [
                0 if i in clsuter_implets_instance_number else 1 for i in range(len(self.test_x))
            ]
            save_dir_cluster_i = os.path.join(save_dir, f'implet{i}')

            result_i, tcav_score_i, cf_score_i = self.probe_shapelet(clsuter_implets, save_dir=save_dir_cluster_i,
                                                                     is_read_pdata=is_read_pdata,
                                                                     shapelet_labels_ori=test_x_shapelet_label,
                                                                     is_model_untrained=is_model_untrained)
            results.append(result_i)
            tcav_scores.append(tcav_score_i)
            cf_scores.append(cf_score_i)
        return results, tcav_scores, cf_scores

    def TCAV_score(self, _probe_coef, _model):
        _fc_weight = _model.fc.weight[0].clone().cpu().detach().numpy()
        _score = _fc_weight @ _probe_coef / np.linalg.norm(_probe_coef) / np.linalg.norm(_fc_weight)
        return _score

    def Latent_CF_score(self, _data_latent, _probe_classifier: sklearn.linear_model.LogisticRegression, _model):
        def project_onto_hyperplane(points, w, b):
            w_norm = np.linalg.norm(w)
            projections = points - ((np.dot(points, w) + b) / (w_norm ** 2))[:, np.newaxis] * w
            return projections

        def flipped_projection(points, w, b, epsilon=0.1):
            """
            Project points onto the hyperplane and then flip their positions to the opposite side.
            """
            # Regular projection
            w_norm = np.linalg.norm(w)
            projections = points - ((np.dot(points, w) + b) / (w_norm ** 2))[:, np.newaxis] * w

            # Determine side of the hyperplane for each point
            signed_distance = np.dot(points, w) + b
            flip_direction = -np.sign(signed_distance)  # Flip the side

            # Modify projection to the opposite side
            flipped_projections = projections + flip_direction[:, np.newaxis] * epsilon * (w / w_norm)
            return flipped_projections

        _probe_coef = _probe_classifier.coef_.flatten()
        _probe_intercept = _probe_classifier.intercept_.flatten()
        latent_CF = flipped_projection(_data_latent, _probe_coef, _probe_intercept)
        CF_result = _model.fc(torch.from_numpy(latent_CF).float().to(self.device))
        CF_pred = np.argmax(CF_result.clone().cpu().detach().numpy(), axis=1)

        test_preds = _model.fc(torch.from_numpy(_data_latent).float().to(self.device))
        test_preds = np.argmax(test_preds.clone().cpu().detach().numpy(), axis=1)
        _score = np.count_nonzero(CF_pred != test_preds) / len(test_preds)
        return _score

    def get_seen_nonfeature(self, inst, attr, k):
        inst = inst.reshape(inst.shape[0], -1)
        attr = attr.reshape(attr.shape[0], -1)
        neg_abs_attr = -np.abs(attr)

        max_sum = -float('inf')
        max_row_idx = -1
        max_start_idx = -1

        # Define a function to compute the sum of a subsequence of length k
        for row_idx in range(neg_abs_attr.shape[0]):
            row = neg_abs_attr[row_idx]

            def subsequence_sum(start_idx):
                return np.sum(row[start_idx:start_idx + k])

            # Create a vectorized version of the subsequence_sum function
            vectorized_subsequence_sum = np.vectorize(subsequence_sum)

            # Generate all valid start indices for subsequences of length k
            valid_indices = np.arange(len(row) - k + 1)

            # Apply the vectorized function to compute the sum for each subsequence
            subsequence_sums = vectorized_subsequence_sum(valid_indices)

            row_max_idx = np.argmax(subsequence_sums)
            row_max_sum = subsequence_sums[row_max_idx]

            # Update the overall minimum subsequence if needed
            if row_max_sum > max_sum:
                max_sum = row_max_sum
                max_row_idx = row_idx
                max_start_idx = valid_indices[row_max_idx]
        # print(max_sum, max_row_idx, max_start_idx, inst[max_row_idx, max_start_idx:max_start_idx + k].shape)
        return inst[max_row_idx, max_start_idx:max_start_idx + k], attr[max_row_idx, max_start_idx:max_start_idx + k]
