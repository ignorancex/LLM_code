from sentence_transformers import SentenceTransformer
import ast
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
from consts import *
import os

class ReviewEncoder:
    def __init__(self, encoder_name, review_dim, sentence_to_vec_model_name=None, force_train=False, save_model=True, save_vectors=True, **training_args):
        self.encoder_name = encoder_name
        if sentence_to_vec_model_name:
            self.sentence_to_vec_model_name = sentence_to_vec_model_name
        self.sentence_to_vec_model = None
        self.review_dim = review_dim
        self.scaler = None
        self.reducer = None
        if not force_train:
            try:
                self.load_encoder()
                self.load_sentence_encoder()
            except FileNotFoundError:
                force_train = True
        if force_train:
            self.load_sentence_encoder()
            self.train_encoder(save_model, save_vectors, training_args)

    def save_metadata(self):
        metadata = {"encoder_name": self.encoder_name,
                    "sentence_to_vec_model_name": self.sentence_to_vec_model_name}
        with open(f"{REVIEW_ENCODERS_PATH}/{self.encoder_name}/metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

    def load_metadata(self):
        with open(f"{REVIEW_ENCODERS_PATH}/{self.encoder_name}/metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        self.sentence_to_vec_model_name = metadata["sentence_to_vec_model_name"]

    def load_sentence_encoder(self):
        print(f"Loading sentence2vec encoder {self.sentence_to_vec_model_name}")
        self.sentence_to_vec_model = SentenceTransformer(self.sentence_to_vec_model_name)
        print("Sentence2vec encoder loaded")

    def reviews2vector(self, reviews):
        reviews_df = pd.DataFrame(reviews).T
        reviews_df['pos_embedding'] = reviews_df['positive'].apply(lambda text: self.sentence_to_vec_model.encode(text).tolist())
        reviews_df['neg_embedding'] = reviews_df['negative'].apply(lambda text: self.sentence_to_vec_model.encode(text).tolist())

        # Generate embeddings for the positive and negative review texts
        reviews_df['pos_embedding'] = reviews_df['positive'].apply(
            lambda text: self.sentence_to_vec_model.encode(text).tolist())
        reviews_df['neg_embedding'] = reviews_df['negative'].apply(
            lambda text: self.sentence_to_vec_model.encode(text).tolist())

        nan_vector = self.sentence_to_vec_model.encode(" ").tolist()
        for sent in ["pos", "neg"]:
            reviews_df[f"{sent}_embedding"] = reviews_df[f"{sent}_embedding"].fillna(str(nan_vector))

        pos_df = pd.DataFrame(
            reviews_df['pos_embedding'].apply(lambda s: ast.literal_eval(s) if isinstance(s, str) else s).tolist(),
            index=reviews_df.index)
        neg_df = pd.DataFrame(
            reviews_df['neg_embedding'].apply(lambda s: ast.literal_eval(s) if isinstance(s, str) else s).tolist(),
            index=reviews_df.index)

        pos_df = pos_df.add_prefix("pos_")
        neg_df = neg_df.add_prefix("neg_")

        vectors = pd.concat([pos_df, neg_df], axis=1)

        return vectors


    def train_encoder(self, save_model, save_vectors, training_args):
        reviews_scores = {}
        DATA_GAME_REVIEWS_PATH = "data/game_reviews"
        reviews_of_hotel = []
        reviews = {}
        hotel_dfs = {}
        for hotel in range(1, 1068 + 1):
            hotel_path = f"{DATA_GAME_REVIEWS_PATH}/{hotel}.csv"
            hotel_csv = pd.read_csv(hotel_path, header=None).fillna("")
            for review in hotel_csv.iterrows():
                reviews[review[1][0]] = {"positive": review[1][2].strip(), "negative": review[1][3].strip(),
                                         "hotel_id": hotel, "score": review[1][4], "hotel_score": hotel_csv[4].mean()}
                reviews_scores[review[1][0]] = review[1][4]
            hotel_dfs[hotel] = hotel_csv
        reviews_df = pd.DataFrame(reviews)

        vectors = self.reviews2vector(reviews_df)

        n_components = self.review_dim
        text_features = vectors
        scaler = StandardScaler()
        text_features_scaled = scaler.fit_transform(text_features)

        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(text_features_scaled)

        if save_vectors:
            principal_df = pd.DataFrame(data=principal_components,
                                        columns=[f'pc_{i}' for i in range(1, n_components + 1)], index=vectors.index)

            principal_df.to_csv(f"{REVIEW_VECTORS_PATH}/{self.encoder_name}.csv")
            print(f"Vectors saved to {REVIEW_VECTORS_PATH}/{self.encoder_name}.csv")

        if save_model:
            self.save_model(scaler, pca)

    def save_model(self, scaler, reducer):
        # create a directory for the encoder
        if not os.path.exists(f"{REVIEW_ENCODERS_PATH}/{self.encoder_name}"):
            os.makedirs(f"{REVIEW_ENCODERS_PATH}/{self.encoder_name}")

        # save metadata
        self.save_metadata()

        # save the scaler and reducer
        with open(f"{REVIEW_ENCODERS_PATH}/{self.encoder_name}/scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)
        with open(f"{REVIEW_ENCODERS_PATH}/{self.encoder_name}/reducer.pkl", 'wb') as f:
            pickle.dump(reducer, f)
        print(f"Model saved to {REVIEW_ENCODERS_PATH}/{self.encoder_name} successfully!")
        self.scaler = scaler
        self.reducer = reducer

    def load_encoder(self):
        # load metadata
        self.load_metadata()
        # load the scaler and reducer
        print("Loading scaler...")
        with open(f"{REVIEW_ENCODERS_PATH}/{self.encoder_name}/scaler.pkl", 'rb') as f:
            self.scaler = pickle.load(f)
        print("Loading reducer...")
        with open(f"{REVIEW_ENCODERS_PATH}/{self.encoder_name}/reducer.pkl", 'rb') as f:
            self.reducer = pickle.load(f)
        print(f"Model loaded from {REVIEW_ENCODERS_PATH}/{self.encoder_name} successfully!")

    def encode(self, pos_part, neg_part):
        if len(pos_part) == 0:
            pos_part = " "
        if len(neg_part) == 0:
            neg_part = " "

        data = pd.DataFrame({"positive": [pos_part], "negative": [neg_part]}).T
        review_vector = self.reviews2vector(data)
        review_vector = self.scaler.transform(review_vector)
        review_vector = self.reducer.transform(review_vector)
        return review_vector

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)


if __name__ == '__main__':
    # Train the model and save the vectors
    sentence_bert = ReviewEncoder("sentence_bert_32", review_dim=32, sentence_to_vec_model_name='all-MiniLM-L6-v2',
                                 train=True, save_model=True, save_vectors=True)

    # Load the model and encode a sentence
    # sentence_bert = ReviewEncoder(encoder_name="sentence_bert_36", sentence_to_vec_model_name='all-MiniLM-L6-v2')
    sent = sentence_bert(pos_part="There is free parking",
                         neg_part="This hotel is the worst I have ever been to. The staff is rude and the rooms are dirty.")
    print(sent)