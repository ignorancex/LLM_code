import gensim

def w2vModel2TXTFile(w2vModelFileName, w2vModelTXTFileName):
	model = gensim.models.Word2Vec.load(w2vModelFileName)
	model.wv.save_word2vec_format(w2vModelTXTFileName)


if __name__ == "__main__":
	w2vModel2TXTFile("arcgis_online_w2v.model", "arcgis_online_w2v.txt",)
