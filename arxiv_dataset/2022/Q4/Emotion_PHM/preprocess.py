import argparse
import os
import re
import pandas as pd
import numpy as np


def preprocess_text(text):
    text = re.sub(r"@\S+", " <user> ", text)  # remove user mentions
    text = re.sub(r"http\S+", " <url> ", text)  # remove
    text = re.sub(r"\s+", " ", text)  # remove double or more space
    text = text.strip()
    return text


def preprocess(dataf, text_column="text"):
    dataf = dataf.copy()

    dataf["text"] = dataf[text_column].apply(preprocess_text)
    return dataf


def save_to_disk(dataf, path, label_column):
    dataf.rename(columns={label_column: "label"}, inplace=True)

    dataf[["text", "label"]].to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="path to data")
    parser.add_argument("--text_column", type=str, help="name of text column header")
    parser.add_argument("--label_column", type=str, help="name of label column header")
    parser.add_argument("--output_dir", type=str, help="path to save processed data")
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)

    preprocessed_df = df.pipe(preprocess, text_column=args.text_column)

    # split to train, validation and test sets
    train, validate, test = np.split(
        preprocessed_df.sample(frac=1, random_state=42),
        [int(0.8 * len(preprocessed_df)), int(0.9 * len(preprocessed_df))],
    )

    # save splits to disk
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    save_to_disk(
        train,
        os.path.join(args.output_dir, "train.csv"),
        label_column=args.label_column,
    )
    save_to_disk(
        validate,
        os.path.join(args.output_dir, "dev.csv"),
        label_column=args.label_column,
    )
    save_to_disk(
        test, os.path.join(args.output_dir, "test.csv"), label_column=args.label_column
    )


if __name__ == "__main__":
    main()
