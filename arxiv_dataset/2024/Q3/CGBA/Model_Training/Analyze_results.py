#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import numpy as np
import json

path = '../COVID_pub'
name = 'COVID'

with open(os.path.join(path, 'fake_clusters.json'), 'r') as file:
    clusters_fake = json.load(file)
with open(os.path.join(path, 'real_clusters.json'), 'r') as file:
    clusters_real = json.load(file)

# Directory containing the results files
results_dir = './PredictResults/Predict_results_dist_alpha0.1_scale0.2_aug10'

# Function to load prediction results from a pickle file
def load_predictions(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

# Function to calculate accuracy from predictions
def calculate_accuracy(predictions, label_ids):
    correct_predictions = (predictions.argmax(axis=1) == label_ids).sum()
    total_samples = len(label_ids)
    return correct_predictions / total_samples if total_samples > 0 else 0

# Function to analyze accuracy for a list of clusters for specified test type
def analyze_accuracy(cluster_list, label, test_type):
    total_samples = 0
    correct_predictions = 0

    for cluster in cluster_list:
        file_path = os.path.join(results_dir, f'{test_type}_cluster{cluster}.pkl')
        if os.path.exists(file_path):
            prediction_output = load_predictions(file_path)
            correct_predictions += (prediction_output.predictions.argmax(axis=1) == prediction_output.label_ids).sum()
            total_samples += len(prediction_output.label_ids)

    overall_accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    print(f"Overall {test_type.replace('_', ' ').capitalize()} Accuracy for {label}: {overall_accuracy:.4f}")

# List containing all clusters
all_clusters = clusters_real + clusters_fake

# Analysis calls

analyze_accuracy(clusters_real, "Real Clusters", "Backdoored_test")
analyze_accuracy(clusters_fake, "Fake Clusters", "Backdoored_test")
analyze_accuracy(all_clusters, "All Clusters", "Backdoored_test")
analyze_accuracy(all_clusters, "All Clusters", "Test")
