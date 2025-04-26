from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


PADDING_RET = '__PAD__'
UNK_RET = '__UNK__'

def init(sentences):
    # lowercase
    tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
    tk.fit_on_texts(sentences)

    raw_text = ' '.join(sentences)

    # create set of chars
    chars = sorted(list(set(raw_text)))
    # chars = set([w_i for w in words for w_i in w])
    n_chars = len(chars)
    print(n_chars)

    # create dictionary of chars and their ids
    char2idx = {c: i + 2 for i, c in enumerate(chars)}
    char2idx["UNK"] = 1
    char2idx["PAD"] = 0

    # Use char_dict to replace the tk.word_index
    tk.word_index = char2idx.copy()

    return tk, char2idx, n_chars

def sentence_split(sentence):
    return sentence.split(' ')


# max_sentence_len max len of sentence in chars
# extracts data with shape (m,max_sentence_len)
# m si number of sentences
def extract_seq_entire_sentence(sentences, tokenizer, max_sentence_len=200):
    data_sequences = tokenizer.texts_to_sequences(sentences)

    # padding
    data = pad_sequences(data_sequences, maxlen=max_sentence_len, padding='post')

    # Convert to numpy array
    data = np.array(data, dtype='float32')

    return data




def pad_list(list_in, max_len, value):
    if len(list_in) > max_len:
        list_in = list_in[:max_len]

    if len(list_in) < max_len:
        sen_len = len(list_in)
        list_in.extend([value for _ in range(max_len - sen_len)])
    return list_in





# extracts
def extract_seq_words(sentences, max_words, max_word_len, char2idx):
    X_char = []
    for sentence in sentences:
        sentence = sentence_split(sentence)
        sent_seq = []
        for i in range(max_words):
            word_seq = []
            for j in range(max_word_len):
                try:
                    word_seq.append(char2idx.get(sentence[i][j]))
                except:
                    word_seq.append(char2idx.get("PAD"))
            sent_seq.append(word_seq)
        X_char.append(np.array(sent_seq))

    return X_char


def init_char_ngram_tokenizers(sentences,ngram_range=(1,2),max_features=101079):
    min_n, max_n = ngram_range
    vectorizers_dict = {}
    for order in range(min_n, max_n+1):
        cvec = CountVectorizer(ngram_range=(order, order), analyzer="char",
                               max_features=max_features)
        cvec = cvec.fit(sentences)
        vectorizers_dict[order] = cvec

    return vectorizers_dict


def utilize_vocab(vocab):
    new_vocab = {}
    for key, value in vocab.items():
        new_vocab[key] = value + 2

    new_vocab[UNK_RET] = 1
    new_vocab[PADDING_RET] = 0

    return new_vocab

def extract_n_gram_seq(sentences, vectorizers_dict, max_sentence_len=200):
    extracted_X_dict = {}

    for ngram_order, vectorizer in vectorizers_dict.items():
        analyzer = vectorizer.build_analyzer()
        vocab = vectorizer.vocabulary_
        vocab = utilize_vocab(vocab)
        # +10 just to be sure
        X_char_sentences = []
        for sentence in sentences:
            tokenized = analyzer(sentence)
            sent_seq = []
            for token in tokenized:
                if vocab.get(token) is None:
                    sent_seq.append(vocab.get(UNK_RET))
                else:
                    try:
                        sent_seq.append(vocab.get(token))
                    except:
                        sent_seq.append(vocab.get(UNK_RET))

            sent_seq = pad_list(sent_seq,max_sentence_len,vocab.get(PADDING_RET))
            # add representation of the sentence
            X_char_sentences.append(np.array(sent_seq))
        # add extracted features fo current ngram
        extracted_X_dict[ngram_order] = X_char_sentences

    return extracted_X_dict




def extract_seq_words_v2(sentences, tokenizer, max_words, max_word_len):
    X_char = []
    for sentence in sentences:
        sentence = sentence_split(sentence)

        sentence_seq = tokenizer.texts_to_sequences(sentence)

        # padd whole sentence - padding words np.full(max_word_len,0) - generate array of len max_words, fi
        sentence_seq = pad_list(sentence_seq, max_words, value=list(np.zeros(max_word_len, dtype=int)))

        # padd word representation
        sentence_seq = pad_sequences(sentence_seq, maxlen=max_word_len, padding='post')

        X_char.append(np.array(sentence_seq))

    return X_char