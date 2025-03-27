





model_file = os.path.join('vector', 'glove_model.txt')
model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=False)

w2v = load_bin_vec(model, vocab)
print('word embeddings loaded!')
print('num words in embeddings: ' + str(len(w2v)))

W, word_idx_map = get_W(w2v, k=model.vector_size)
x_train, x_dev, x_test = make_idx_data(X_train, X_dev, X_test, word_idx_map, maxlen=36)
max_features = W.shape[0]