import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler
import pickle
from sklearn.decomposition import KernelPCA
import numpy as np
import sys
sys.path.append('..')
from library.py import compute_pca_metrics_and_svm

df = pd.read_csv('framingham.csv')
df = df.drop(['education'], axis=1)
df.dropna(inplace=True)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
sampler = RandomOverSampler(sampling_strategy=0.6)
X_resampled, y_resampled = sampler.fit_resample(X, y)
X_train, _, y_train, _ = train_test_split(X_resampled, y_resampled, train_size=0.8, stratify=y_resampled, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
print("Evaluating SVM without PCA:")
svm = SVC(kernel='linear')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(svm, X_scaled, y_train, cv=skf, scoring='accuracy')
mean_score = scores.mean()
std_dev = scores.std()
print(f'Cross-validated accuracy without PCA: {mean_score:.4f} ± {std_dev:.4f}')
optimal_components = None
best_score = -float('inf')
print("Determining optimal number of components using linear kernel PCA and linear SVM:")
for n_components in range(2, 8):
    kpca = KernelPCA(n_components=n_components, kernel='linear')
    X_kpca = kpca.fit_transform(X_scaled)
    print(f'\nRunning SVM with Kernel PCA ({n_components} components)')
    svm = SVC(kernel='linear')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(svm, X_kpca, y_train, cv=skf, scoring='accuracy')
    mean_score = scores.mean()
    print(f'Cross-validated accuracy with {n_components} components: {mean_score:.4f} ± {scores.std():.4f}')
    if mean_score > best_score:
        best_score = mean_score
        optimal_components = n_components
print(f'\nOptimal number of components determined: {optimal_components} with accuracy: {best_score:.4f}')

kernel_files = [
    'heart_kernel_matrix_noise_level_0.pkl',
    'heart_kernel_matrix_noise_level_1.pkl',
    'heart_kernel_matrix_noise_level_2.pkl'
]

for kernel_file in kernel_files:
    with open(kernel_file, 'rb') as file:
        precomputed_kernel = pickle.load(file)   
    print(f'\nEvaluating kernel file: {kernel_file}')
    print(f'Kernel matrix size: {precomputed_kernel.shape}')
    print(f'Kernel matrix (first 5 elements):\n{precomputed_kernel[:5, :5]}')
    print(f'Variance within the kernel matrix: {np.var(precomputed_kernel)}')
    print('\nRunning custom kernel SVM directly with precomputed kernel')
    svm_custom = SVC(kernel='precomputed')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores_custom_direct = cross_val_score(svm_custom, precomputed_kernel, y_train, cv=skf, scoring='accuracy')
    print(f'Custom kernel SVM direct accuracy: {scores_custom_direct.mean():.4f} ± {scores_custom_direct.std():.4f}')
    print("\nEvaluating CPV, Reconstruction Error, and SVM Accuracy with different number of components:")
    n_components_list = range(2, 8)
    metrics = compute_pca_metrics_and_svm(precomputed_kernel, y_train, n_components_list)
    for n_components, metric in metrics.items():
        print(f'\nNumber of components: {n_components}')
        print(f'CPV: {metric["CPV"]:.4f}')
        print(f'Reconstruction Error: {metric["Reconstruction Error"]:.4f}')
        print(f'SVM Accuracy: {metric["SVM Accuracy"]:.4f} ± {metric["SVM Accuracy Std"]:.4f}')
