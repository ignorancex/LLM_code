import fasttext

def load_ft_model():
    model_file = '../Data/wiki/wiki.en.bin'
    return fasttext.load_model(model_file)

def get_query_embeddings(ft_model,q):
    return ft_model.get_sentence_vector(q)

