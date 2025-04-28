"""
Helper + analysis + visualization functions common to most stages of the pipeline, along with important
imports

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import numpy as np
import os
import joblib
import shutil
import pandas as pd
import SimpleITK as sitk

def scale_feature_importances(feat_imp):
    """
    Function to scale feature importances between 0 and 1
    For a proper comparison across seeds

    """
    max_value = np.amax(feat_imp)
    min_value = np.amin(feat_imp)

    scaled_feat_imp =  (feat_imp - min_value)/(max_value-min_value)
    return scaled_feat_imp


def create_features_matrix(df, selected_features=None):

    features_only = df.drop(['Patient ID', 'slice', 'label'], axis=1)
    feature_names = features_only.columns.to_numpy()

    # Drop features
    if selected_features is not None:
        features_to_drop = list(set(list(feature_names))-set(selected_features))
        features_only = features_only.drop(labels=features_to_drop, axis=1)
        feature_names = features_only.columns.to_numpy()

    features_np = features_only.to_numpy()
    labels_array = df['label'].to_numpy()

    return features_np, labels_array, feature_names

def get_model_path(out_dir=None, reduce_features=False, oversample=False):
    model_name = 'model'
    if reduce_features is True:
        model_name = model_name + '_red_feat'
    if oversample is True:
        model_name = model_name + '_oversample'

    model_path = os.path.join(out_dir, '{}.pkl'.format(model_name))
    return model_path

def normalize_features(features_matrix, mean=None, std=None):

    if mean is None and std is None:
        mean = np.mean(features_matrix, axis=0)
        std = np.std(features_matrix, axis=0)

    norm_feat = (features_matrix-mean)/(std+1e-5)
    return norm_feat, mean, std

#def save_itk_image(image, fname):
#    """
#    Save a 3-D ITK image
#
#    Image dims = H x W x D or C x H x W x D
#
#    """
#    assert(isinstance(image, np.ndarray))
#    if image.ndim == 3:
#        image = np.transpose(image, (2, 0, 1))
#        image_itk = sitk.GetImageFromArray(image)
#    elif image.ndim == 4:
#        image_channels = []
#        n_channels = image.shape[0]
#        for channel in range(n_channels):
#            image_channels.append(sitk.GetImageFromArray(image[channel, :, :, :].transpose(2, 0, 1)))
#        image_itk = sitk.JoinSeries(image_channels)
#
#    sitk.WriteImage(image_itk, fname)
#

def convert_patch_to_itk(patch, direction, origin, spacing):
    assert(isinstance(patch, np.ndarray))
    patch_itk = sitk.GetImageFromArray(patch)
    patch_itk.SetDirection(direction)
    patch_itk.SetOrigin(origin)
    patch_itk.SetSpacing(spacing)
    return patch_itk

#def perform_PCA(dataset):
#    assert(isinstance(dataset, np.ndarray))
#    assert(dataset.ndim == 2)
#    n_features = dataset.shape[1]
#    # Keep all features
#    pca = PCA(n_components=n_features)
#    pca = pca.fit(dataset)
#    return pca.transform(dataset), pca
#
#def create_feature_matrix_from_list(features_ds):
#    """
#    Used to create data matrix to train classifer after
#    splitting on a per-patient basis
#
#    """
#    assert(isinstance(features_ds, np.ndarray))
#    features_matrix = []
#    label_array = []
#    for idx in range(features_ds.shape[0]):
#        per_pat_dict = features_ds[idx]
#
#        for feature_vec, data in zip(per_pat_dict['features'], per_pat_dict['data']):
#            features_matrix.append(feature_vec)
#            label_array.append(data[1])
#
#    features_matrix = np.array(features_matrix)
#    label_array = np.array(label_array)
#    return features_matrix, label_array
#
#def save_model(classifier, save_dir=None, fname=None):
#    if os.path.exists(save_dir) is False:
#        os.makedirs(save_dir)
#
#    assert(fname is not None)
#
#    fname = os.path.join(save_dir, fname)
#    joblib.dump(classifier, fname)
#
#
#def load_model(save_dir=None, fname=None):
#    assert(os.path.exists(save_dir))
#    assert(fname is not None)
#    if os.path.exists(os.path.join(save_dir, fname)) is False:
#        return None
#
#    cls = joblib.load(os.path.join(save_dir, fname))
#    return cls
#
#def plot_decision_boundary(norm_data_matrix, label_array, svm_classifier, fname=None):
#    """
#    Visualization function to plot decision boundary for linear SVM
#
#    pipeline is tuple containing the following blocks:
#        0: data scaler
#        1: pca
#        2: pca transformed data scaler
#        3: svm classifier
#
#    """
#
#    # Even though the problem is solved using the dual representation,
#    # coef_ returns the weight coefficients from the primal formulation
#    weights = svm_classifier.best_estimator_.coef_
#    bias = svm_classifier.best_estimator_.intercept_
#    assert(weights.shape[1] == norm_data_matrix.shape[1]) # Sanity check on dim and primal SVM
#
#    tp_features = []
#    fp_features = []
#
#    for idx, label in enumerate(label_array):
#        features = norm_data_matrix[idx]
#        if label == 0:
#            tp_features.append(features)
#        else:
#            fp_features.append(features)
#
#    tp_features = np.array(tp_features)
#    fp_features = np.array(fp_features)
#    #fig = plt.figure()
#    ax = plt.subplot(111, projection='3d')
#    ax.scatter(tp_features[:, 0], tp_features[:, 1], tp_features[:, 2], c='tab:blue', label='True Positives')
#    ax.scatter(fp_features[:, 0], fp_features[:, 1], fp_features[:, 2], c='tab:orange', label='False Positives')
#    ax.legend()
#    ax.set_xlabel('Feature 0')
#    ax.set_ylabel('Feature 1')
#    ax.set_zlabel('Feature 2')
#    ax.set_title('SVM decision boundary')
#    ax.grid(True)
#
#    # Plot the SVM decision boundary
#    xlist = np.linspace(-2, 2, 200)
#    ylist = np.linspace(-2, 2, 200)
#    X, Y = np.meshgrid(xlist, ylist)
#    # The hyperplane equation is <w, X> + b = 0 (X is the feature vector)
#    # So we can write one co-ordinate in terms of the other 2
#    Z = -(weights[0,0]*X + weights[0, 1]*Y + bias)/weights[0, 2]
#    ax.plot_surface(X, Y, Z)
#
#    #fig.savefig(fname)
#    #plt.close()
#    return plt
#
#
#def plot_decision_regions(norm_data_matrix, label_array, classifier, title=None, axes_labels=None):
#
#    # Even though the problem is solved using the dual representation,
#    # coef_ returns the weight coefficients from the primal formulation
#    #weights = svm_classifier.best_estimator_.coef_
#    #bias = svm_classifier.best_estimator_.intercept_
#    #assert(weights.shape[1] == norm_data_matrix.shape[1]) # Sanity check on dim and primal SVM
#
#    # Plot the SVM decision boundary
#
#
#    x_min, x_max = norm_data_matrix[:, 0].min() - 1, norm_data_matrix[:, 0].max() + 1
#    y_min, y_max = norm_data_matrix[:, 1].min() - 1, norm_data_matrix[:, 1].max() + 1
#
#    xlist = np.arange(x_min, x_max, 0.02)
#    ylist = np.arange(y_min, y_max, 0.02)
#
#    xx, yy = np.meshgrid(xlist, ylist)
#
#    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
#    Z = Z.reshape(xx.shape)
#
#    ax = plt.subplot(111)
#    ax.contourf(xx, yy, Z, cmap=plt.cm.Paired)
#
#    # Plot the actual data points
#    tp_features = []
#    fp_features = []
#
#    for idx, label in enumerate(label_array):
#        features = norm_data_matrix[idx]
#        if label == 0:
#            tp_features.append(features)
#        else:
#            fp_features.append(features)
#
#    tp_features = np.array(tp_features)
#    fp_features = np.array(fp_features)
#
#    ax.scatter(tp_features[:, 0], tp_features[:, 1], c='tab:blue')
#    ax.scatter(fp_features[:, 0], fp_features[:, 1], c='tab:orange')
#
#    if axes_labels is None:
#        ax.set_xlabel('Feature 0')
#        ax.set_ylabel('Feature 1')
#    else:
#        ax.set_xlabel(axes_labels[0])
#        ax.set_ylabel(axes_labels[1])
#
#
#    if title is None:
#        ax.set_title('SVM decision surface')
#    else:
#        ax.set_title(title)
#
#    return plt
#
#def eigen_analysis(feature_matrix=None, pca=None):
#
#    # Tranposed to get DxN shape (see MML book by Diesenroth et al.)
#    if feature_matrix is not None:
#        trans_feat_matrix = np.transpose(feature_matrix)
#        cov_matrix = np.cov(trans_feat_matrix)
#        eigvals, eigvecs = np.linalg.eig(cov_matrix)
#        print('Eigen-values of the covariance matrix:')
#        print(eigvals)
#
#        print('Eigen-vectors of the covariance matrix:')
#        print(eigvecs)
#
#
#
#def plot_features(features_matrix, label_array, fname=None, title=None, axes_labels=()):
#
#    tp_features = []
#    fp_features = []
#    for idx, label in enumerate(label_array):
#        features = features_matrix[idx]
#        if label == 0:
#            tp_features.append(features)
#        else:
#            fp_features.append(features)
#
#    tp_features = np.array(tp_features)
#    fp_features = np.array(fp_features)
#
#    if features_matrix.shape[1] == 3:
#        ax = plt.subplot(111, projection='3d')
#        ax.scatter(tp_features[:, 0], tp_features[:, 1], tp_features[:, 2], c='tab:blue', label='True Positives')
#        ax.scatter(fp_features[:, 0], fp_features[:, 1], fp_features[:, 2], c='tab:orange', label='False Positives')
#        if len(axes_labels) == 0:
#            ax.set_xlabel('Feature 0')
#            ax.set_ylabel('Feature 1')
#            ax.set_zlabel('Feature 2')
#        else:
#            assert(len(axes_labels) == 3)
#            ax.set_xlabel(axes_labels[0])
#            ax.set_ylabel(axes_labels[1])
#            ax.set_zlabel(axes_labels[2])
#    else:
#        ax = plt.subplot(111)
#        ax.scatter(tp_features[:, 0], tp_features[:, 1], c='tab:blue', label='True Positives')
#        ax.scatter(fp_features[:, 0], fp_features[:, 1], c='tab:orange', label='False Positives')
#        if len(axes_labels) == 0:
#            ax.set_xlabel('Feature 0')
#            ax.set_ylabel('Feature 1')
#        else:
#            assert(len(axes_labels) == 2)
#            ax.set_xlabel(axes_labels[0])
#            ax.set_ylabel(axes_labels[1])
#
#    ax.legend()
#
#    if title is None:
#        ax.set_title('Data scatter plot')
#    else:
#        ax.set_title(title)
#
#    ax.grid(True)
#
#    return plt
#
#def calculate_mean_std(scores):
#
#    if isinstance(scores, list) is True:
#        scores = np.array(scores)
#
#    assert(scores.ndim == 1)
#
#    mean = np.mean(scores)
#    std = np.std(scores)
#    return mean, std
#
#def get_mode_from_path(path):
#
#    mode = path.split('/')[-1]
#    if mode == '':
#        mode = path.split('/')[-2]
#    return mode
