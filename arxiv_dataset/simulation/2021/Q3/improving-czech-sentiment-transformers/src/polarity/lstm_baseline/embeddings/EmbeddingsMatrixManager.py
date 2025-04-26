import numpy as np
import errno
import os
# import gzip
from src.polarity.lstm_baseline.nn_utils import save_data_pickle, load_data_pickle


class EmbeddingsMatrixManager:
    def __init__(self, embeddings_filename, cached_embeddings_filename,
                 we_embedding_matrix_filename, wordmap_filename, dimension,
                 max_words=None, use_gzip=True):

        # file with embeddings
        self.embeddings_filename = embeddings_filename

        # file with extracted we matrix, indices are numbers
        self.we_embedding_matrix_filename = we_embedding_matrix_filename

        # file with cached embedings - indices are words
        self.cached_embeddings_filename = cached_embeddings_filename

        # embeddings dimension
        self.dimension = dimension

        # cached word map
        self.wordmap_filename = wordmap_filename

        # use gzip for embedding file loading
        self.use_gzip = use_gzip

        # maximum of words
        self.max_words = max_words

    def create_we_matrix(self):
        # https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

        if os.path.exists(self.embeddings_filename):
            print('Creating we matrix from file {}'.format(self.embeddings_filename))
            embeddings_word_indices = {}

            if self.use_gzip:
                # f = gzip.open(self.embeddings_filename, 'rt',encoding='utf-8')
                f = None
                pass
            else:
                f = open(self.embeddings_filename, 'r', encoding='utf-8')

            # first line contains size of vocabulary and dimension
            try:
                n, d = map(int, f.readline().split())
                print('Word vectors with voc:', n, " and dimension:", d)
            except ValueError:
                print('Invalid format of the embeddings')

            for i, line in enumerate(f):
                # if max words is set
                if self.max_words is not None:
                    # +1 because we keep one space for unknown
                    if (i + 1) >= self.max_words:
                        print("Loading done, reached max vectors:" + str(i+1) + " keeping one place for unk token")
                        break

                values = line.split()
                word = values[0]
                if (len(values)) != (self.dimension + 1):
                    print('Invalid line parsing:', line, end='')
                    continue
                try:
                    coefs = np.asarray(values[1:], dtype='float16')
                    embeddings_word_indices[word] = coefs
                except ValueError:
                    print('Error during parsing line:', line)

                if (i % 100000 == 0):
                    print('Processed ', i, ' vectors')

            f.close()
            print('Found %s word vectors.' % len(embeddings_word_indices))

            # save created matrix
            save_data_pickle(embeddings_word_indices, self.cached_embeddings_filename)
            print('we matrix saved')

        else:
            print("{} not found!".format(self.embeddings_filename))
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.embeddings_filename)

    def get_we_matrix(self):
        '''
        create matrix with word embeddings indexed with numbers
        :return: we matrix and map word map (mapping word to indices)
        '''

        if os.path.exists(self.wordmap_filename) and os.path.exists(self.we_embedding_matrix_filename):
            we_matrix = load_data_pickle(self.we_embedding_matrix_filename)
            word_map = load_data_pickle(self.wordmap_filename)
        else:
            vectors = self.read_embeddings_matrix()
            vocab_size = len(vectors)
            print('Loaded %s word vectors.' % vocab_size)

            # indices for words to we matrix
            word_map = {}
            pos = 0
            # +1 for zero padding token and +1 for unk
            we_matrix = np.ndarray((vocab_size + 2, self.dimension), dtype='float16')

            for i, (word, vector) in enumerate(vectors.items()):
                pos = i + 1
                word_map[word] = pos
                we_matrix[pos] = vector

            # add unknown token
            pos += 1
            word_map["<unk>"] = pos
            we_matrix[pos] = np.random.uniform(low=-0.20, high=0.20, size=self.dimension)

            save_data_pickle(we_matrix, self.we_embedding_matrix_filename)
            save_data_pickle(word_map, self.wordmap_filename)

        # index 0 - only zeros
        return we_matrix, word_map

    def read_embeddings_matrix(self):
        if os.path.exists(self.cached_embeddings_filename):
            return load_data_pickle(self.cached_embeddings_filename)
        else:
            self.create_we_matrix()
            return self.read_embeddings_matrix()
