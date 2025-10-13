#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__description__ = (
    "calculate TGF-IDF for FSoLS-25-v5_fulltext datatset with nltk "
    "separated in two steps due to long time of processing"
    "this is step 1 ending up with TF-csv for all four categories "
    "step two: calculate_tf-idf_step-2.py"
)
__author__ = "Eva Seidlmayer <seidlmayer@zbmed.de>"
__copyright__ = "2025 by Eva Seidlmayer"
__license__ = "MIT license"
__email__ = "seidlmayer@zbmed.de"
__version__ = "1.0.1 "


from fontTools.colorLib.builder import populateCOLRv0
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import argparse
import nltk


def load_dataset(input_file_csv):
    # Load dataset

    df = pd.read_csv(input_file_csv, sep=",", usecols=["text", "category_id"])
    df = df.replace(r"^\s*$", np.nan, regex=True)
    # df = df.dropna()
    df = df.sample(frac=1)
    df = df.astype(str)
    science_df = df[df["category_id"] == "scientific"]
    vernacular_df = df[df["category_id"] == "popular"]
    alt_science_df = df[df["category_id"] == "alternative_science"]
    disinfo_df = df[df["category_id"] == "disinfo"]

    science_texts = science_df["text"].to_list()
    vernacular_texts = vernacular_df["text"].to_list()
    alt_science_texts = alt_science_df["text"].to_list()
    disinfo_texts = disinfo_df["text"].to_list()
    print("data input lists created")

    science_tokenz = [word_tokenize(sentence) for sentence in science_texts]
    vernacular_tokenz = [word_tokenize(sentence) for sentence in vernacular_texts]
    alt_science_tokenz = [word_tokenize(sentence) for sentence in alt_science_texts]
    disinfo_tokenz = [word_tokenize(sentence) for sentence in disinfo_texts]
    print("tokenz created")

    science_flat_list = [item for sublist in science_tokenz for item in sublist]
    vernacular_flat_list = [item for sublist in vernacular_tokenz for item in sublist]
    alt_science_flat_list = [item for sublist in alt_science_tokenz for item in sublist]
    disinfo_flat_list = [item for sublist in disinfo_tokenz for item in sublist]
    print("flat lists created")

    return (
        science_flat_list,
        vernacular_flat_list,
        alt_science_flat_list,
        disinfo_flat_list,
    )


def prepare_text(name, item):
    # load stopwords:
    stops = stopwords.words("english")
    # transform all word characters to lower case:
    list_of_words = [word.lower() for word in item]
    # remove all words containing up to two characters:
    list_of_words = [word for word in list_of_words if len(word) > 2]
    # remove stopwords:
    list_of_words = [word for word in list_of_words if word not in stops]
    return list_of_words


def count_freq(category):
    unique_words = []
    counts = []
    # create a list of unique words:
    for item in category:
        if item not in unique_words:
            unique_words.append(item)
    # count the frequency of each word:
    for word in unique_words:
        count = 0
        for i in category:
            if word == i:
                count += 1
        counts.append(count)
    df = pd.DataFrame({"word": unique_words, "count": counts})
    df.sort_values(by="count", inplace=True, ascending=False)
    df.reset_index(drop=True, inplace=True)
    print(
        "xxxxxxxxxxxxxxxxxxxxx next category xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    )
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file_csv")
    parser.add_argument("output_file_csv")
    args = parser.parse_args()

    # load dataset
    (
        science_flat_list,
        vernacular_flat_list,
        alt_science_flat_list,
        disinfo_flat_list,
    ) = load_dataset(args.input_file_csv)

    class_words = {
        "science_flat_list": science_flat_list,
        "vernacular_flat_list": vernacular_flat_list,
        "alt_science_flat_list": alt_science_flat_list,
        "disinfo_flat_list": disinfo_flat_list,
    }

    prepared_texts = {}

    for name, item in class_words.items():
        # print('name', name, 'item', item)
        texts_prepared = prepare_text(name, item)
        prepared_texts[f"{name}_prepared"] = texts_prepared
    print("texts prepared")

    for name, category in prepared_texts.items():
        category_text_df = count_freq(category)
        print(f"counted frequency of {name}")

        category_text_df["TF"] = category_text_df["count"] / sum(
            category_text_df["count"]
        )
        output_file_name = args.output_file_csv +  f'{name}'
        category_text_df.to_csv(output_file_name),
            index=False,
        )
        print("calculated TF of", name)


if __name__ == "__main__":
    main()
