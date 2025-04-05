import numpy as np
import pandas as pd


def filter_senses(senses: pd.DataFrame, sense_labels: pd.DataFrame) -> pd.DataFrame:
    """
    Remove senses that are not used in dataset images.

    Args:
        senses: A dataframe of verb senses
        sense_labels: A dataframe containing the correct sense for each
            pair (image, verb).

    Returns:
        A dataset containing only the senses in 'sense_labels'
    """
    new_senses = pd.DataFrame(columns=['lemma', 'sense_num', 'definition',
                                       'ontonotes_sense_examples', 'visualness_label'])
    for _, row in enumerate(senses.itertuples()):
        sense = getattr(row, 'lemma')
        sense_id = getattr(row, 'sense_num')

        occurrences = sense_labels.query("lemma == @sense and sense_chosen == @sense_id")
        if occurrences.shape[0] > 0:
            new_senses = new_senses.append([row], sort=False)
    new_senses = new_senses.drop(columns=['Index'])
    new_senses.reset_index(inplace=True, drop=True)
    return new_senses


def filter_image_name(img_name: str) -> str:
    """
    Remove image name prefixes.

    In COCO image annotations labels had a prefix which is incompatible
    with other image names sources. The purpose of this function is
    to remove such prefixes.

    Args:
        img_name: image name in the form PREFIX_XXXX.jpeg

    Returns:
        The XXXX image identifier

    Raises:
        ValueError: when the image prefix is not known
    """
    train_prefix = 'COCO_train2014_'
    val_prefix = 'COCO_val2014_'
    if img_name.startswith(train_prefix):
        stripped_zeros = train_prefix + str(int(img_name[len(train_prefix):-4]))
    elif img_name.startswith(val_prefix):
        stripped_zeros = val_prefix + str(int(img_name[len(val_prefix):-4]))
    else:
        stripped_zeros = img_name
    return stripped_zeros.split('.')[0]


def combine_data(embeddings: pd.DataFrame, images_features: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate the 300-dim word-embeddings-vector and the 4096-dim
    VGG16 feature vector and unit-normalise the output vector.

    Args:
        embeddings: embedding vector
        images_features: visual feature-vector

    Returns:
        A dataframe containing the columns:
            'e_caption', 'e_object', 'e_combined', 'e_image',
            'concat_image_caption', 'concat_image_object', 'concat_image_text'.
    """
    full_dataframe = pd.concat([embeddings, images_features], axis=1, sort=True)
    full_dataframe['concat_image_caption'] = full_dataframe.apply(
        lambda r: np.concatenate([r.e_caption, r.e_image.ravel()]), axis=1)
    full_dataframe['concat_image_object'] = full_dataframe.apply(
        lambda r: np.concatenate([r.e_object, r.e_image.ravel()]), axis=1)
    full_dataframe['concat_image_text'] = full_dataframe.apply(
        lambda r: np.concatenate([r.e_combined, r.e_image.ravel()]), axis=1)
    return full_dataframe.applymap(lambda x: x / np.linalg.norm(x, ord=2))


def aggregate_stats(experiments_path):
    columns = ['labels_per_class', 'alpha', 'verb_type', 'representation_type', 'accuracy']
    experiments = pd.read_csv(experiments_path, names=columns)[['labels_per_class', 'representation_type', 'verb_type', 'accuracy']]
    aggregated_data = experiments.groupby(['labels_per_class', 'representation_type', 'verb_type'], as_index=False).agg({'accuracy': ['mean', lambda x: x.std(ddof=0)]})
    aggregated_data.columns = ['labels_per_class', 'representation_type', 'verb_type', 'mean', 'std']
    aggregated_data.to_html('results.html')
    print('Aggregated results written to an HTML file.')

