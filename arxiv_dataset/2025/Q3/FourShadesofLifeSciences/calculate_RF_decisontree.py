#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__description__ = "print decision tree of random forrest"
__author__ = "Eva Seidlmayer <seidlmayer@zbmed.de>"
__copyright__ = "2024-2025 by Eva Seidlmayer"
__license__ = "MIT license"
__email__ = "seidlmayer@zbmed.de"
__version__ = "1.0.1 "

import numpy as np
import argparse
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import export_text, plot_tree
from sklearn.tree import DecisionTreeClassifier



plt.rcParams.update({"figure.figsize": (12.0, 8.0)})
plt.rcParams.update({"font.size": 14})


def load_dataset(input_file_csv):
    df = pd.read_csv(input_file_csv, sep=",", usecols=["text", "category_id"])
    df = df.replace(r"^\s*$", np.nan, regex=True)
    labels = df["category_id"].to_list()
    df["text"] = df["text"].fillna("")
    category_counts = Counter(labels)
    print("category counts:", category_counts)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file_csv")
    args = parser.parse_args()


    df = load_dataset(args.input_file_csv)

    # Preprocess text data (convert text to numerical features using TF-IDF)
    vectorizer = TfidfVectorizer()
    texts = vectorizer.fit_transform(df["text"])

    # Encode the target labels (convert categories to numbers)
    encoder = LabelEncoder()
    labels = encoder.fit_transform(df["category_id"])

    # Split the data into training and testing sets
    texts_train, texts_test, labels_train, labels_test = train_test_split(
        texts, labels, random_state=0
    )

    # Random Forests in `scikit-learn` (with N = 100)
    rf = RandomForestClassifier(n_estimators=100, random_state=0)

    rf.fit(texts_train, labels_train)

    for number in range(100):
        estimator = rf.estimators_[number]
        tree_rules = export_text(
            estimator, feature_names=vectorizer.get_feature_names_out()
        )
        path = args.output
        text_file = open(
            f"{path}{number}.text",
            "w",
        )
        text_file.write(tree_rules)

    # Optional: Print tree rules
    tree_rules = export_text(
        estimator, feature_names=vectorizer.get_feature_names_out()
    )

    dtclf = DecisionTreeClassifier(max_depth=4)
    dtclf = dtclf.fit(texts_train, labels_train)
    print('feature importance:', dtclf.feature_importances_)
    print('done')


if __name__ == "__main__":
    main()
