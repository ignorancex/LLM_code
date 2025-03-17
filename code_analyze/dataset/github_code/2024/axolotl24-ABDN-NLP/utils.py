import pickle
import os
import numpy as np
import nltk
from nltk.corpus import stopwords

def all_files(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            yield fullname


def find_files(dir):
    files = []
    for file in all_files(dir):
        # convert to linux file path
        file = '/'.join(file.split('\\'))
        files.append(file)
    return files

def create_filepath_dir(filepath):
    dir = os.path.dirname(filepath)
    if not os.path.exists(dir):
        os.makedirs(dir)

def load_text_corpus(file_path):
    with open(file_path, "r", encoding="utf-8") as corpus_data:
        corpus_raw = corpus_data.read().split('\n')
    return corpus_raw

def save_to_disk(pickle_f, obj):
    dir = os.path.dirname(pickle_f)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(pickle_f, "wb") as f:
        pickle.dump(obj, f)

def load_from_disk(pickle_f):
    if not os.path.exists(pickle_f):
        print("pickle file not exists: {}".format(pickle_f))
        return None
    with open(pickle_f, "rb") as f:
        obj = pickle.load(f)
    return obj

def save_as_txt(path, obj):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(path, "w") as f:
        f.write(str(obj))


def save_sendata_as_txt(path, sendata):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    data_str = ''
    for sen in sendata:
        data_str += sen
        data_str += '\n'
    with open(path, "w", encoding="utf-8") as f:
        f.write(data_str)


def update_static_embeds(static_embeds, stat_emb):
    index_voc, voc_index, embeddings = static_embeds
    idxs = [key for key in index_voc]
    voc_len = np.max(idxs) if len(idxs)>0 else 0
    voc_len += 1
    for word in stat_emb:
        if word not in voc_index:
            voc_index[word] = voc_len
            index_voc[voc_len] = word
            embeddings[voc_len] = stat_emb[word]
            voc_len += 1
        else:
            embeddings[voc_index[word]] = \
                (embeddings[voc_index[word]] + stat_emb[word])/2
    static_embeds = (index_voc, voc_index, embeddings)

    return static_embeds


def init_static_embedding():
    index_voc = {}
    voc_index = {}
    embeddings = {}
    static_embeds = (index_voc, voc_index, embeddings)
    return static_embeds


def merge_static_embeds(static_embeds1, static_embeds2):
    static_embeds = init_static_embedding()
    index_voc, voc_index, embeddings = static_embeds
    index_voc1, voc_index1, embeddings1 = static_embeds1
    index_voc2, voc_index2, embeddings2 = static_embeds2
    voc_len = 0
    for word in voc_index1:
        if word not in voc_index:
            voc_index[word] = voc_len
            index_voc[voc_len] = word
            embeddings[voc_len] = embeddings1[voc_index1[word]]
            voc_len += 1

    for word in voc_index2:
        if word not in voc_index:
            voc_index[word] = voc_len
            index_voc[voc_len] = word
            embeddings[voc_len] = embeddings2[voc_index2[word]]
            voc_len += 1
        else:
            embeddings[voc_index[word]] = \
                (embeddings[voc_index[word]] + embeddings2[voc_index2[word]])/2

    return static_embeds

def exists(f):
    return os.path.exists(f)

def stop_words_en(language='en'):
    support_languages = {'en': "./stopwords/stop_words_english-small.txt"}
    if language not in support_languages:
        print("Unsupported stop word for language ", language)
        return []
    with open(support_languages['en'], "r", encoding="utf-8") as f:
        swl = f.read().split('\n')
    return swl

def stop_words_de():
    return stopwords.words('german')

def stop_words_sv():
    return stopwords.words('swedish')

def stop_words_la(language='la'):
    support_languages = {'la': "./stopwords/stop_words_latin.txt"}
    if language not in support_languages:
        print("Unsupported stop word for language ", language)
        return []
    with open(support_languages['la'], "r", encoding="utf-8") as f:
        swl = f.read().split(',')
    return swl


def stop_words_ru():
    dir = './nltk_stopwords'
    if not os.path.exists(dir):
        print("download stopwords to ", dir)
        nltk.download('stopwords', dir)
    return stopwords.words('russian')


def stop_words_fi():
    dir = './nltk_stopwords'
    if not os.path.exists(dir):
        print("download stopwords to ", dir)
        nltk.download('stopwords', dir)
    return stopwords.words('finnish')


def stop_words(language=None):
    support_languages = {'en': stop_words_en,
                         'de': stop_words_de,
                         'sv': stop_words_sv,
                         'la': stop_words_la,
                         'ru': stop_words_ru,
                         'fi': stop_words_fi,}
    if language not in support_languages:
        print("Unsupported stop word for language ", language)
        return []
    fun = support_languages[language]
    swl = fun()
    return swl

def word_voc(word, voc_index, embeddings):
    id = voc_index[word]
    return embeddings[id]

def cosine_similarity(v_w1, v_w2):
    theta_sum = np.dot(v_w1, v_w2)
    theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
    theta = theta_sum / theta_den
    return theta


def word_cosine_similarity(w1, w2, voc_index, embeddings):
    v_w1 = word_voc(w1, voc_index, embeddings)
    v_w2 = word_voc(w2, voc_index, embeddings)
    theta = cosine_similarity(v_w1, v_w2)
    return theta


def most_sim_words(v_w1, top_n, voc_index, embeddings, willprint=True):
    word_sim = {}
    for word_c in voc_index:
        v_w2 = word_voc(word_c, voc_index, embeddings)
        theta = cosine_similarity(v_w1, v_w2)
        word_sim[word_c] = theta
    words_sorted = sorted(word_sim.items(), key=lambda kv: kv[1], reverse=True)
    words = []
    for word, sim in words_sorted[:top_n]:
        if willprint:
            print(word, sim)
        words.append((word, sim, word_voc(word, voc_index, embeddings)))

    return words




