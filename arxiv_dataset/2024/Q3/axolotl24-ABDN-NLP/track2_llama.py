import argparse
import pathlib
import pandas as pd
import torch
import tqdm
from prompt_llama import assessment_def_block
import warnings

warnings.filterwarnings("ignore")

# helper function to remove unusable datapoints
def remove_unusable(dataset):
    pre_drop = len(dataset)
    dataset = dataset.dropna(subset=["example", "word"])
    post_drop = len(dataset)
    if pre_drop != post_drop:
        n_dropped = pre_drop - post_drop
        print(
            f"[Warn] some datapoints were dropped {n_dropped} / {pre_drop}, i.e. "
            f"{(n_dropped * 100) / pre_drop:.2f}%"
        )
    return dataset

def do_prompt_to_def(dataset, language, lm):

    dataset = remove_unusable(dataset)
    old_senses = set(dataset[dataset.period == "old"].sense_id.unique())
    dataset = dataset[~dataset.sense_id.apply(old_senses.__contains__)]

    dataset["predicted gloss"] = ''
    for sense_id in tqdm.tqdm(dataset.sense_id.unique()):
        sense_data = dataset[dataset.sense_id == sense_id]
        if language == 'Finnish':
            headwords = [example[int(pos.split(';')[0].split(':')[0]):int(pos.split(';')[0].split(':')[1])] for pos, example in zip(sense_data.indices_target_token.tolist(), sense_data.example.tolist())]
        else:
            headwords = None
        sentences = sense_data.example.tolist()
        target_word = sense_data.word.tolist()[0]

        gloss = assessment_def_block(
            target_word,
            sentences,
            language,
            headwords,
            lm)

        print('definition', gloss)

        row_number = sense_data.index
        dataset.loc[row_number, "predicted gloss"] = gloss

    dataset[["sense_id", "word", "predicted gloss"]].rename(
        columns={"predicted gloss": "gloss"}
    ).to_csv(args.predictions_file, sep="\t")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_data",
        type=pathlib.Path,
        help="path to test data.",
        required=True,
    )
    parser.add_argument(
        "--predictions_file",
        type=pathlib.Path,
        help="path to prediction output file.",
        default="preds.tsv",
    )

    parser.add_argument("--lang", help="Langauge", type=str, default='ru')

    args = parser.parse_args()

    test_dataset = pd.read_csv(args.test_data, sep="\t")

    if args.lang=='ru':
      language = 'Russian'
    if args.lang == 'fi':
      language = 'Finnish'
    if args.lang == 'de':
      language = 'German'

    do_prompt_to_def(test_dataset, language, lm=None)
