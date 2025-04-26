import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from pathlib import Path

import logging
import os

from config import LOGGING_FORMAT, LOGGING_DATE_FORMAT, FACEBOOK_ORIGINAL_DIR, FACEBOOK_DATASET_DIR, RANDOM_SEED, \
    FACEBOOK_DATASET_TRAIN, FACEBOOK_DATASET_TEST, FACEBOOK_DATASET_DEV, FACEBOOK_DATASET, \
    CSFD_ORIGINAL_DIR, CSFD_DATASET, CSFD_DATASET_DIR, CSFD_DATASET_TRAIN, CSFD_DATASET_TEST, CSFD_DATASET_DEV, \
    MALLCZ_ORIGINAL_DIR, MALL_DATASET, MALL_DATASET_DIR, MALL_DATASET_TRAIN, MALL_DATASET_TEST, MALL_DATASET_DEV, \
    CSFD_TEST_SIZE, CSFD_DEV_SIZE, FACEBOOK_TEST_SIZE, FACEBOOK_DEV_SIZE, MALLCZ_TEST_SIZE, MALLCZ_DEV_SIZE, \
    IMDB_ORIGINAL_DIR, IMDB_CL_TEST_SIZE, IMDB_CL_DEV_SIZE, IMDB_DATASET_CL_DIR, IMDB_DATASET_CL, IMDB_DATASET_CL_TRAIN, \
    IMDB_DATASET_CL_TEST, IMDB_DATASET_CL_DEV, IMDB_DATASET, IMDB_DATASET_DIR, IMDB_DATASET_TRAIN, IMDB_DATASET_TEST, \
    IMDB_DATASET_DEV
from src.polarity.data.loader import DATASET_LOADERS

logging.basicConfig(format=LOGGING_FORMAT,
                    datefmt=LOGGING_DATE_FORMAT)
logger = logging.getLogger(__name__)

COLOR_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(COLOR_PALETTE))

""""
Script for splitting Czech sentiment data
"""


def main():
    split_imdb_original()
    split_imdb_crosslingual()

    split_facebook()
    split_csfd()
    split_mall()

    print_datset_info(DATASET_LOADERS['csfd'](-1,False), 'CSFD')
    print_datset_info(DATASET_LOADERS['mallcz'](-1,False), 'Mallcz')
    print_datset_info(DATASET_LOADERS['fb'](-1,False), 'FB')

def print_datset_info(dataset_loader, dataset_name):
    print("="*70)
    print("Printing stats for dataset:" + dataset_name)
    print("Total results:")
    dataset_loader.load_data()
    df_entire = dataset_loader.load_entire_dataset()
    entire_counts = df_entire.label_text.value_counts()
    print(entire_counts)
    print("Total:" + str(len(df_entire)))

    print("------")
    print("Train results:")
    df_train = dataset_loader.get_train_data()
    train_counts = df_train.label_text.value_counts()
    print(train_counts)
    print("Total:" + str(len(df_train)))

    print("------")
    print("Test results:")
    df_test = dataset_loader.get_test_data()
    test_counts = df_test.label_text.value_counts()
    print(test_counts)
    print("Total:" + str(len(df_test)))

    print("------")
    print("Dev results:")
    df_dev = dataset_loader.get_dev_data()
    dev_counts = df_dev.label_text.value_counts()
    print(dev_counts)
    print("Total:" + str(len(df_dev)))

    pass

# this prepares the original dataset into our format
def split_imdb_original():
    logger.info("Loading IMDB original dataset")
    df_train = load_imdb_polarity(os.path.join(IMDB_ORIGINAL_DIR, 'train'))
    df_test = load_imdb_polarity(os.path.join(IMDB_ORIGINAL_DIR, 'test'))

    print("Train")
    print(f"df shape{df_train.shape}")
    print(f"Size:{len(df_train)}")
    print("Counts:")
    print(df_train['label_text'].value_counts())
    print(df_train.head())

    print("\nTest")
    print(f"df shape{df_test.shape}")
    print(f"Size:{len(df_test)}")
    print("Counts:")
    print(df_test['label_text'].value_counts())
    print(df_test.head())

    Path(os.path.join(IMDB_DATASET_DIR)).mkdir(parents=True, exist_ok=True)
    entire_dataset_df = pd.concat([df_train, df_test], axis=0)
    entire_dataset_df.reset_index(drop=True, inplace=True)
    # save entire dataset
    entire_dataset_df.to_csv(IMDB_DATASET, encoding='utf-8', index=False)
    entire_dataset_df.reset_index(drop=True, inplace=True)

    distribut_path = os.path.join(IMDB_DATASET_DIR, 'dataset_distribution.png')
    logger.info(f"Saving dataset distribution image{distribut_path}")

    figure = sns.countplot(entire_dataset_df.label_text).get_figure()
    figure.savefig(distribut_path, dpi=400)
    plt.figure()

    train_dir = os.path.join(IMDB_DATASET_DIR, 'train')
    test_dir = os.path.join(IMDB_DATASET_DIR, 'test')
    dev_dir = os.path.join(IMDB_DATASET_DIR, 'dev')

    Path(os.path.join(train_dir)).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(test_dir)).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(dev_dir)).mkdir(parents=True, exist_ok=True)

    # shuffle train dataset
    df_train = shuffle(df_train, random_state=RANDOM_SEED)
    df_train.reset_index(drop=True, inplace=True)

    # shuffle test dataset - it is not necessary to shuffle it but whatever..
    df_test = shuffle(df_test, random_state=RANDOM_SEED)
    df_test.reset_index(drop=True, inplace=True)

    # there are no dev data in the dataset
    df_dev = pd.DataFrame(columns=['text', 'label', 'label_text'])

    # stats
    print("Counts Train:")
    print(df_train['label_text'].value_counts())
    print(df_train.head())
    print(70 * "-")

    print("Counts Dev:")
    print(df_dev['label_text'].value_counts())
    print(df_dev.head())
    print(70 * "-")

    print("Counts Test:")
    print(df_test['label_text'].value_counts())
    print(df_test.head())
    print(70 * "-")

    # save distribution
    train_distr_path = os.path.join(IMDB_DATASET_DIR, 'train', 'train_distribution.png')
    test_distr_path = os.path.join(IMDB_DATASET_DIR, 'test', 'test_distribution.png')
    dev_distr_path = os.path.join(IMDB_DATASET_DIR, 'dev', 'dev_distribution.png')

    figure_train = sns.countplot(df_train.label_text).get_figure()
    figure_train.savefig(train_distr_path, dpi=400)
    plt.figure()  # it resets the plot

    figure_test = sns.countplot(df_test.label_text).get_figure()
    figure_test.savefig(test_distr_path, dpi=400)
    plt.figure()  # it resets the plot

    if len(df_dev) > 0:
        figure_dev = sns.countplot(df_dev.label_text).get_figure()
        figure_dev.savefig(dev_distr_path, dpi=400)
        plt.figure()  # it resets the plot

    # save datasets
    df_train.to_csv(IMDB_DATASET_TRAIN, encoding='utf-8', index=False)
    df_test.to_csv(IMDB_DATASET_TEST, encoding='utf-8', index=False)
    df_dev.to_csv(IMDB_DATASET_DEV, encoding='utf-8', index=False)



# this is split used for training english -> czech crosslingual model
# we split the dataset for crosslingual experiments so that wee use 90% for train and 10% for eval
def split_imdb_crosslingual():
    logger.info("Loading IMDB  dataset")
    df = load_imdb_polarity_entire_original(IMDB_ORIGINAL_DIR)
    logger.info("IMDB Dataset loaded")

    print(f"df shape{df.shape}")
    print(f"Size:{len(df)}")
    print("Counts:")
    print(df['label_text'].value_counts())
    print(df.head())

    Path(os.path.join(IMDB_DATASET_CL_DIR)).mkdir(parents=True, exist_ok=True)
    # save entire dataset
    df.to_csv(IMDB_DATASET_CL, encoding='utf-8', index=False)
    df.reset_index(drop=True, inplace=True)

    distribut_path = os.path.join(IMDB_DATASET_CL_DIR, 'dataset_distribution.png')
    logger.info(f"Saving dataset distribution image{distribut_path}")

    figure = sns.countplot(df.label_text).get_figure()
    figure.savefig(distribut_path, dpi=400)
    plt.figure()

    train_dir = os.path.join(IMDB_DATASET_CL_DIR, 'train')
    test_dir = os.path.join(IMDB_DATASET_CL_DIR, 'test')
    dev_dir = os.path.join(IMDB_DATASET_CL_DIR, 'dev')

    Path(os.path.join(train_dir)).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(test_dir)).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(dev_dir)).mkdir(parents=True, exist_ok=True)

    # shuffle and split
    df = shuffle(df, random_state=RANDOM_SEED)
    df.reset_index(drop=True, inplace=True)
    df_train, df_test = train_test_split(df, test_size=IMDB_CL_TEST_SIZE, random_state=RANDOM_SEED)
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    if IMDB_CL_DEV_SIZE == 0:
        df_dev = pd.DataFrame(columns=['text', 'label', 'label_text'])
    else:
        df_train, df_dev = train_test_split(df_train, test_size=IMDB_CL_DEV_SIZE, random_state=RANDOM_SEED)
        df_dev.reset_index(drop=True, inplace=True)
        df_train.reset_index(drop=True, inplace=True)

    # stats
    print("Counts Train:")
    print(df_train['label_text'].value_counts())
    print(df_train.head())
    print(70 * "-")

    print("Counts Dev:")
    print(df_dev['label_text'].value_counts())
    print(df_dev.head())
    print(70 * "-")

    print("Counts Test:")
    print(df_test['label_text'].value_counts())
    print(df_test.head())
    print(70 * "-")

    # save distribution
    train_distr_path = os.path.join(IMDB_DATASET_CL_DIR, 'train', 'train_distribution.png')
    test_distr_path = os.path.join(IMDB_DATASET_CL_DIR, 'test', 'test_distribution.png')
    dev_distr_path = os.path.join(IMDB_DATASET_CL_DIR, 'dev', 'dev_distribution.png')

    figure_train = sns.countplot(df_train.label_text).get_figure()
    figure_train.savefig(train_distr_path, dpi=400)
    plt.figure()  # it resets the plot

    figure_test = sns.countplot(df_test.label_text).get_figure()
    figure_test.savefig(test_distr_path, dpi=400)
    plt.figure()  # it resets the plot

    if len(df_dev) > 0:
        figure_dev = sns.countplot(df_dev.label_text).get_figure()
        figure_dev.savefig(dev_distr_path, dpi=400)
        plt.figure()  # it resets the plot

    # save datasets
    df_train.to_csv(IMDB_DATASET_CL_TRAIN, encoding='utf-8', index=False)
    df_test.to_csv(IMDB_DATASET_CL_TEST, encoding='utf-8', index=False)
    df_dev.to_csv(IMDB_DATASET_CL_DEV, encoding='utf-8', index=False)


# -------------------------Mall----------------------
def split_mall():
    logger.info("Loading MALL CZ dataset")
    df = load_mallcz_polarity_original(MALLCZ_ORIGINAL_DIR)
    logger.info("MALL CZ Dataset loaded")
    print(f"df shape{df.shape}")
    print(f"Size:{len(df)}")
    print("Counts:")
    print(df['label_text'].value_counts())
    print(df.head())

    Path(os.path.join(MALL_DATASET_DIR)).mkdir(parents=True, exist_ok=True)
    # save entire dataset
    df.to_csv(MALL_DATASET, encoding='utf-8', index=False)
    df.reset_index(drop=True, inplace=True)

    distribut_path = os.path.join(MALL_DATASET_DIR, 'dataset_distribution.png')
    logger.info(f"Saving dataset distribution image{distribut_path}")

    figure = sns.countplot(df.label_text).get_figure()
    figure.savefig(distribut_path, dpi=400)
    plt.figure()

    train_dir = os.path.join(MALL_DATASET_DIR, 'train')
    test_dir = os.path.join(MALL_DATASET_DIR, 'test')
    dev_dir = os.path.join(MALL_DATASET_DIR, 'dev')

    Path(os.path.join(train_dir)).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(test_dir)).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(dev_dir)).mkdir(parents=True, exist_ok=True)

    # shuffle and split
    df = shuffle(df, random_state=RANDOM_SEED)
    df.reset_index(drop=True, inplace=True)
    df_train, df_test = train_test_split(df, test_size=MALLCZ_TEST_SIZE, random_state=RANDOM_SEED)
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    df_train, df_dev = train_test_split(df_train, test_size=MALLCZ_DEV_SIZE, random_state=RANDOM_SEED)
    df_dev.reset_index(drop=True, inplace=True)
    df_train.reset_index(drop=True, inplace=True)

    # stats
    print("Counts Train:")
    print(df_train['label_text'].value_counts())
    print(df_train.head())
    print(70 * "-")

    print("Counts Dev:")
    print(df_dev['label_text'].value_counts())
    print(df_dev.head())
    print(70 * "-")

    print("Counts Test:")
    print(df_test['label_text'].value_counts())
    print(df_test.head())
    print(70 * "-")

    # save distribution
    train_distr_path = os.path.join(MALL_DATASET_DIR, 'train', 'train_distribution.png')
    test_distr_path = os.path.join(MALL_DATASET_DIR, 'test', 'test_distribution.png')
    dev_distr_path = os.path.join(MALL_DATASET_DIR, 'dev', 'dev_distribution.png')

    figure_train = sns.countplot(df_train.label_text).get_figure()
    figure_train.savefig(train_distr_path, dpi=400)
    plt.figure()  # it resets the plot

    figure_test = sns.countplot(df_test.label_text).get_figure()
    figure_test.savefig(test_distr_path, dpi=400)
    plt.figure()  # it resets the plot

    figure_dev = sns.countplot(df_dev.label_text).get_figure()
    figure_dev.savefig(dev_distr_path, dpi=400)
    plt.figure()  # it resets the plot

    # save datasets
    df_train.to_csv(MALL_DATASET_TRAIN, encoding='utf-8', index=False)
    df_test.to_csv(MALL_DATASET_TEST, encoding='utf-8', index=False)
    df_dev.to_csv(MALL_DATASET_DEV, encoding='utf-8', index=False)


# -------------------------CSFD----------------------
def split_csfd():
    logger.info("Loading CSFD dataset")
    df = load_csfd_polarity_original(CSFD_ORIGINAL_DIR)
    logger.info("CSFD Dataset loaded")
    print(f"df shape{df.shape}")
    print(f"Size:{len(df)}")
    print("Counts:")
    print(df['label_text'].value_counts())
    print(df.head())

    Path(os.path.join(CSFD_DATASET_DIR)).mkdir(parents=True, exist_ok=True)
    # save entire dataset
    df.to_csv(CSFD_DATASET, encoding='utf-8', index=False)
    df.reset_index(drop=True, inplace=True)

    distribut_path = os.path.join(CSFD_DATASET_DIR, 'dataset_distribution.png')
    logger.info(f"Saving dataset distribution image{distribut_path}")

    figure = sns.countplot(df.label_text).get_figure()
    figure.savefig(distribut_path, dpi=400)
    plt.figure()

    train_dir = os.path.join(CSFD_DATASET_DIR, 'train')
    test_dir = os.path.join(CSFD_DATASET_DIR, 'test')
    dev_dir = os.path.join(CSFD_DATASET_DIR, 'dev')

    Path(os.path.join(train_dir)).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(test_dir)).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(dev_dir)).mkdir(parents=True, exist_ok=True)

    # shuffle and split
    df = shuffle(df, random_state=RANDOM_SEED)
    df.reset_index(drop=True, inplace=True)
    df_train, df_test = train_test_split(df, test_size=CSFD_TEST_SIZE, random_state=RANDOM_SEED)
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    df_train, df_dev = train_test_split(df_train, test_size=CSFD_DEV_SIZE, random_state=RANDOM_SEED)
    df_dev.reset_index(drop=True, inplace=True)
    df_train.reset_index(drop=True, inplace=True)

    # stats
    print("Counts Train:")
    print(df_train['label_text'].value_counts())
    print(df_train.head())
    print(70 * "-")

    print("Counts Dev:")
    print(df_dev['label_text'].value_counts())
    print(df_dev.head())
    print(70 * "-")

    print("Counts Test:")
    print(df_test['label_text'].value_counts())
    print(df_test.head())
    print(70 * "-")

    # save distribution
    train_distr_path = os.path.join(CSFD_DATASET_DIR, 'train', 'train_distribution.png')
    test_distr_path = os.path.join(CSFD_DATASET_DIR, 'test', 'test_distribution.png')
    dev_distr_path = os.path.join(CSFD_DATASET_DIR, 'dev', 'dev_distribution.png')

    figure_train = sns.countplot(df_train.label_text).get_figure()
    figure_train.savefig(train_distr_path, dpi=400)
    plt.figure()  # it resets the plot

    figure_test = sns.countplot(df_test.label_text).get_figure()
    figure_test.savefig(test_distr_path, dpi=400)
    plt.figure()  # it resets the plot

    figure_dev = sns.countplot(df_dev.label_text).get_figure()
    figure_dev.savefig(dev_distr_path, dpi=400)
    plt.figure()  # it resets the plot

    # save datasets
    df_train.to_csv(CSFD_DATASET_TRAIN, encoding='utf-8', index=False)
    df_test.to_csv(CSFD_DATASET_TEST, encoding='utf-8', index=False)
    df_dev.to_csv(CSFD_DATASET_DEV, encoding='utf-8', index=False)


# -------------------------Facebook----------------------
def split_facebook():
    logger.info("Loading facebook dataset")
    df = load_fb_polarity_original(FACEBOOK_ORIGINAL_DIR)
    logger.info("Facebook dataset loaded")
    print(f"df shape{df.shape}")
    print(f"Size:{len(df)}")
    print("Counts:")
    print(df['label_text'].value_counts())
    print(df.head())

    Path(os.path.join(FACEBOOK_DATASET_DIR)).mkdir(parents=True, exist_ok=True)
    # save entire dataset
    df.to_csv(FACEBOOK_DATASET, encoding='utf-8', index=False)

    distribut_path = os.path.join(FACEBOOK_DATASET_DIR, 'dataset_distribution.png')
    logger.info(f"Saving dataset distribution image{distribut_path}")

    figure = sns.countplot(df.label_text).get_figure()
    figure.savefig(distribut_path, dpi=400)
    plt.figure()

    train_dir = os.path.join(FACEBOOK_DATASET_DIR, 'train')
    test_dir = os.path.join(FACEBOOK_DATASET_DIR, 'test')
    dev_dir = os.path.join(FACEBOOK_DATASET_DIR, 'dev')

    Path(os.path.join(train_dir)).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(test_dir)).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(dev_dir)).mkdir(parents=True, exist_ok=True)

    # shuffle and split
    df = shuffle(df, random_state=RANDOM_SEED)
    df.reset_index(drop=True, inplace=True)
    df_train, df_test = train_test_split(df, test_size=FACEBOOK_TEST_SIZE, random_state=RANDOM_SEED)
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    df_train, df_dev = train_test_split(df_train, test_size=FACEBOOK_DEV_SIZE, random_state=RANDOM_SEED)
    df_dev.reset_index(drop=True, inplace=True)
    df_train.reset_index(drop=True, inplace=True)

    # stats
    print("Counts Train:")
    print(df_train['label_text'].value_counts())
    print(df_train.head())
    print(70 * "-")

    print("Counts Dev:")
    print(df_dev['label_text'].value_counts())
    print(df_dev.head())
    print(70 * "-")

    print("Counts Test:")
    print(df_test['label_text'].value_counts())
    print(df_test.head())
    print(70 * "-")

    # save distribution
    train_distr_path = os.path.join(FACEBOOK_DATASET_DIR, 'train', 'train_distribution.png')
    test_distr_path = os.path.join(FACEBOOK_DATASET_DIR, 'test', 'test_distribution.png')
    dev_distr_path = os.path.join(FACEBOOK_DATASET_DIR, 'dev', 'dev_distribution.png')

    figure_train = sns.countplot(df_train.label_text).get_figure()
    figure_train.savefig(train_distr_path, dpi=400)
    plt.figure()  # it resets the plot

    figure_test = sns.countplot(df_test.label_text).get_figure()
    figure_test.savefig(test_distr_path, dpi=400)
    plt.figure()  # it resets the plot

    figure_dev = sns.countplot(df_dev.label_text).get_figure()
    figure_dev.savefig(dev_distr_path, dpi=400)
    plt.figure()  # it resets the plot

    # save datasets
    df_train.to_csv(FACEBOOK_DATASET_TRAIN, encoding='utf-8', index=False)
    df_test.to_csv(FACEBOOK_DATASET_TEST, encoding='utf-8', index=False)
    df_dev.to_csv(FACEBOOK_DATASET_DEV, encoding='utf-8', index=False)

def load_imdb_polarity(imdb_dir):
    pos_dir = (1, 'positive', os.path.join(imdb_dir, 'pos'))
    neg_dir = (0, 'negative', os.path.join(imdb_dir, 'neg'))

    all_data_dirs = [pos_dir, neg_dir]

    return process_imdb(all_data_dirs)



def load_imdb_polarity_entire_original(imdb_dir):
    train_pos_dir = (1, 'positive', os.path.join(imdb_dir, 'train', 'pos'))
    train_neg_dir = (0, 'negative', os.path.join(imdb_dir, 'train', 'neg'))
    test_pos_dir = (1, 'positive', os.path.join(imdb_dir, 'test', 'pos'))
    test_neg_dir = (0, 'negative', os.path.join(imdb_dir, 'test', 'neg'))

    all_data_dirs = [train_pos_dir, train_neg_dir, test_pos_dir, test_neg_dir]
    return process_imdb(all_data_dirs)


def process_imdb(all_data_dirs):
    all_data_files = []

    for label_int, label, tmp_dir in all_data_dirs:
        logger.info("Searching files from:" + str(tmp_dir))
        tmp_files = [(label_int, label, os.path.join(tmp_dir, f)) for f in os.listdir(tmp_dir) if os.path.isfile(os.path.join(tmp_dir, f))]
        logger.info("Found num of files:" + str(len(tmp_files)))
        all_data_files.extend( tmp_files)

    logger.info("Total of files found:" + str(len(all_data_files)))

    logger.info("Loading data")

    data_tmp = []
    for i, (label_int, label, tmp_file) in enumerate(all_data_files):
        text = ''
        with open(tmp_file, 'r', encoding='utf-8') as file:
            text = file.read().replace('\n', ' ')
            text = text.replace('<br /><br />', ' ')
            text = text.replace('<br />', ' ')
            text = text.strip()

        if not text:
            logger.info("The text is empty:")

        # data_tmp.append([text, label_int, label, tmp_file])
        data_tmp.append([text, label_int, label])
        if i % 5000 == 0 and i > 0:
            logger.info("Loaded:" + str(i))

    # data_df = pd.DataFrame(data_tmp, columns=['text', 'label', 'label_text', 'file_name'])
    data_df = pd.DataFrame(data_tmp, columns=['text', 'label', 'label_text'])
    logger.info("Total loaded:" + str(len(data_df)))

    return data_df





def load_mallcz_polarity_original(mallcz_dir):
    df_negative = pd.read_csv(os.path.join(mallcz_dir, "negative.txt"), names=["text"], sep="\r\n", engine='python')
    df_negative['label'] = 0
    df_negative['label_text'] = 'negative'

    df_positive = pd.read_csv(os.path.join(mallcz_dir, "positive.txt"), names=["text"], sep="\r\n", engine='python')
    df_positive['label'] = 1
    df_positive['label_text'] = 'positive'

    df_neutral = pd.read_csv(os.path.join(mallcz_dir, "neutral.txt"), names=["text"], sep="\r\n", engine='python')
    df_neutral['label'] = 2
    df_neutral['label_text'] = 'neutral'

    # create one dataframe
    df = pd.concat([df_negative, df_positive, df_neutral], axis=0)

    return df


def load_csfd_polarity_original(csfd_dir):
    df_negative = pd.read_csv(os.path.join(csfd_dir, "negative.txt"), names=["text"], sep="\r\n", engine='python')
    df_negative['label'] = 0
    df_negative['label_text'] = 'negative'

    df_positive = pd.read_csv(os.path.join(csfd_dir, "positive.txt"), names=["text"], sep="\r\n", engine='python')
    df_positive['label'] = 1
    df_positive['label_text'] = 'positive'

    df_neutral = pd.read_csv(os.path.join(csfd_dir, "neutral.txt"), names=["text"], sep="\r\n", engine='python')
    df_neutral['label'] = 2
    df_neutral['label_text'] = 'neutral'

    # create one dataframe
    df = pd.concat([df_negative, df_positive, df_neutral], axis=0)

    return df


# Vse v jedne funkci
def load_fb_polarity_original(fb_dir, drop_both=True):
    # load separate files
    df_labels = pd.read_csv(os.path.join(fb_dir, "gold-labels.txt"), names=["label"])
    df_posts = pd.read_csv(os.path.join(fb_dir, "gold-posts.txt"), sep="\r\n", names=["text"], engine='python')

    # add one more column for text label
    df_labels_text = df_labels.copy()
    df_labels_text.columns = ['label_text']

    # create one dataframe
    df = pd.concat([df_posts, df_labels, df_labels_text], axis=1)

    # replace label with numbers, and label_text with text
    df['label'] = df.label.apply(replace_label)
    df['label_text'] = df.label_text.apply(replace_label_text)

    # drop both examples
    if drop_both is True:
        df = df[df.label != 3]

    return df

# replace label with a number
# 0 - negative
# 1 - positive
# 2 - neutral
# 3 - b (both)
def replace_label(label):
  if label == 'n':
    return 0
  elif label == 'p':
    return 1
  elif label == '0':
    return 2
  else:
    return 3

def replace_label_text(label):
  if label == 'n':
    return "negative"
  elif label == 'p':
    return "positive"
  elif label == '0':
    return "neutral"
  else:
    return "both"


if __name__ == '__main__':
    logging.root.setLevel(level=logging.INFO)
    main()