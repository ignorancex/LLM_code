"""Embed Text

Captions and Senses are preprocessed (typos fix, tokenization, stopwords
filtering) and embedded to a dataframe
(Caption, Object class, Caption+Object class).
"""
import numpy as np
import pandas as pd
import gensim.downloader as api
from gensim.utils import simple_preprocess
from nltk.corpus import wordnet as wn
from stop_words import get_stop_words
from features_extraction.typos import TYPOS, TYPOS2


def preprocess_text(caption, model, stop_words, word_net_categories=False):
    """
    Tokenize and remove stopwords/unknown words from a string.
    If word_net_categories is true, and a word is not in the dictionary, its
    WordNet ancestor is returned until the term is known (used for object labels
    embedding).

    Args:
        caption: input string
        model: word2vec model
        stop_words: list of stopwords
        word_net_categories: whether to traverse WordNet hierarchy for unknown
            words or not

    Returns:
        A list of filtered words
    """
    caption_tokens = [w for w in simple_preprocess(caption) if not w in stop_words]
    if word_net_categories:
        new_tokens = []
        for token in caption_tokens:
            if token in model.vocab:
                new_tokens.append(token)
            else:
                new_token = check_synset_ancestors(token, model)
                if new_token != '':
                    new_tokens.append(new_token)
        return new_tokens
    return [token for token in caption_tokens if token in model.vocab]


def embed_text(text_tokens, model):
    """
    Embed a sequence of words and average and normalise them.

    Args:
        text_tokens: tokenised string
        model: trained word2vec model

    Returns:
        A 300-dim normalised numpy vector
    """
    if text_tokens == [] or text_tokens is None:
        return None
    word_average = np.mean([model.word_vec(token) for token in text_tokens], 0)
    return word_average / np.linalg.norm(word_average, ord=2)


def embed_data_descriptions(model, input_df):
    """
    Embed image descriptions into 300-dim vectors using word2vec
    embedding.

    Args:
        model: pre-trained word2vec gensim model
        input_df: dataframe of image captions with columns:
            image_id, object, caption

    Returns:
        A dataframe of numpy 300-dim vectors with image id as index
        and columns: 'e_caption' (embedded caption), 'e_object'
        (embedded object), 'e_combined' (combination of caption and
        object embedding).
    """

    # Stopwords definition
    stop_words = list(get_stop_words('en'))

    concat_strings = lambda x: "%s" % ', '.join(x)
    grouped_captions = input_df.groupby('image_id')['caption'].unique().apply(
        lambda x: concat_strings(x[:3]))
    grouped_categories = input_df.groupby('image_id')['object'].unique().apply(concat_strings)
    descriptions_df = pd.concat([grouped_captions, grouped_categories], axis=1)

    # Caption preprocessing
    captions_tokens = descriptions_df['caption'].apply(
        lambda r: preprocess_text(r, model, stop_words))
    categories_tokens = descriptions_df['object'].apply(
        lambda r: preprocess_text(r, model, stop_words, True))

    # Caption embedding
    descriptions_df['e_caption'] = captions_tokens.apply(lambda r: embed_text(r, model))
    descriptions_df['e_object'] = categories_tokens.apply(lambda r: embed_text(r, model))
    descriptions_df['e_combined'] = (captions_tokens + categories_tokens).apply(
        lambda r: embed_text(r, model))

    return descriptions_df.drop(['caption', 'object'], axis=1)


def embed_data_senses(model, input_df):
    """
    Embed verb senses into 300-dim vectors using word2vec embedding.

    Args:
        model: pre-trained word2vec gensim model
        input_df: dataframe of senses definitions with columns:
            lemma, sense_num, definition, ontonotes_sense_examples

    Returns:
        A dataframe of numpy 300-dim vectors with image id as index
        and columns: 'e_definition' (embedded definition), 'e_examples'
        (embedded examples), 'e_combined' (combination of definition and
        examples embedding), 'lemma', 'sense_num'.
    """
    # Stopwords definition
    stop_words = list(get_stop_words('en'))

    senses_df = input_df.dropna().reset_index(drop=True)

    # Senses preprocessing
    definitions_tokens = senses_df['definition'].apply(
        lambda r: preprocess_text(r, model, stop_words))
    examples_tokens = senses_df['ontonotes_sense_examples'].apply(
        lambda r: preprocess_text(r.replace('\n', ' '), model, stop_words))

    # Senses embedding
    combined_tokens = definitions_tokens + examples_tokens
    senses_df['e_definition'] = definitions_tokens.apply(lambda r: embed_text(r, model))
    senses_df['e_examples'] = examples_tokens.apply(lambda r: embed_text(r, model))
    senses_df['e_combined'] = combined_tokens.apply(lambda r: embed_text(r, model))

    senses_df.drop(
        ['definition', 'ontonotes_sense_examples', 'visualness_label'], axis=1, inplace=True)

    return senses_df


def spell_fix(filepath, typos):
    """
    Correct 'filepath' text using 'typos'.
    Write the fixed test in a file _filepath

    Args:
        filepath: path of file to fix
        typos: list of tuples ('wrong', 'correct')

    Returns:
        None
    """
    path_parts = filepath.split('/')
    path_parts[-1] = '_' + path_parts[-1]
    new_filepath = 'generated/' + path_parts[-1]

    with open(filepath, 'r') as file:
        data = file.read()
        for typo, correct in typos:
            data = data.replace(typo, correct)
    with open(new_filepath, 'w') as file:
        file.write(data)


def check_synset_ancestors(word, model):
    """
    Return the first ancestor in the WordNet synset hierarchy of the
    target word which is in the model dictionary.

    Args:
        word: target word to encode whose ancestor must be extracted
        model: embedding model

    Returns:
        An ancestor of the target word which is available in the
        dictionary
    """
    synset = wn.synsets(word)
    if synset == []:
        return ""
    synset = synset[0]
    tree = synset.tree(lambda s: s.hypernyms())
    if len(tree) < 2:
        return ""
    branch = tree[1]

    while True:
        if len(tree) < 2:
            raise ValueError('Can\' find ancestor in synset tree')
        token = branch[0].name().split('.')[0]
        if token not in model.vocab:
            branch = branch[1]
        else:
            return token


def main():
    """
    Load data, spellcheck and embed.
    """
    print('Loading word2vec Network...')
    model = api.load('word2vec-google-news-300')
    model.init_sims(replace=True)

    print('Spellchecking annotations...')
    spell_fix('generated/verse_annotations.csv', TYPOS)
    print('Embedding VerSe annotations...')
    verse_embedding = embed_data_descriptions(
        model, pd.read_csv('generated/_verse_annotations.csv'))
    print('Writing Data...')
    verse_embedding.to_pickle('generated/verse_embedding.pkl')
    del verse_embedding

    print('Spellchecking senses...')
    spell_fix('../data/labels/verse_visualness_labels.tsv', TYPOS2)
    print('Embedding senses...')
    embedded_senses = embed_data_senses(model, pd.read_csv('generated/_verse_visualness_labels.tsv',
                                                           sep='\t', dtype={'sense_num': str}))
    print('Writing Data...')
    embedded_senses.to_pickle('generated/senses_embedding.pkl')
    del embedded_senses


if __name__ == '__main__':
    main()
