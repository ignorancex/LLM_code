"""
For licensing see accompanying LICENSE file.
Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""

import time
import pandas as pd
import torch
from transformers.utils import logging
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import vec2text
import pickle
from utils.sparse_autoencoder import SparseAutoencoder, load_sae_file
from helpers import correct_multiple_sentences, correct_sentence, format_new_points_interpolate, format_new_points_umap, generate_prompt_ideas, generate_sentence_variations, mean_pool, format_new_points, prompt_for_sentence_variations_llm
import os
import umap

logging.set_verbosity_error()  # Suppress warnings

# SETTINGS
model_folder = "../models/"
data_folder = "../data/"
activation_threshold = 0.01

# SAE CLASS
os.environ["TOKENIZERS_PARALLELISM"] = "false"

wiki_sae_datasets = ['wiki', 'writing', 'chi']


class SAE(object):
    def __init__(self, model_dict, dataset="wiki"):
        print(f'initializing {dataset} SAE...')
        self.device = model_dict['device']

        # get dataset name without numbers
        dataset_no_num = ''.join(
            [i for i in dataset if not i.isdigit()]) if dataset not in wiki_sae_datasets else 'wiki'

        # Load the sae model
        sae_path = model_folder + f'{dataset_no_num}_model.safetensors'
        self.sae = SparseAutoencoder.from_safetensors_file(
            sae_path) if dataset_no_num == 'wiki' else load_sae_file(sae_path)
        self.sae.to(self.device)  # move the model to the device
        print('\nsae model loaded')

        # Load the embedding model and tokenizer
        self.embedding_model = model_dict['embedding_model']
        self.tokenizer = model_dict['tokenizer']
        self.corrector = model_dict['corrector']

        # Load the features
        features_file = model_folder + \
            f'{dataset_no_num}_features.npy'
        self.features = np.load(features_file)
        self.feature_sim_matrix = cosine_similarity(
            self.features)  # Precompute the similarity matrix
        features_info_file = model_folder + \
            f'{dataset_no_num}_feature_info.csv'
        self.feature_info = pd.read_csv(features_info_file)
        print('\nfeatures loaded')
        print('features shape:', self.features.shape)
        print('feature_info shape:', self.feature_info.shape)

        # Load the umap model
        umap_file = model_folder + f'{dataset}_umap_reducer'
        umap_pickle = open(umap_file, 'rb')
        self.umap_reducer = pickle.load(umap_pickle)
        print('\numap reducer loaded')

        emb_file = data_folder + f"{dataset}/{dataset}_embeddings.pt"
        self.embeddings = torch.load(
            emb_file, map_location=self.device, weights_only=True)
        self.embed_sim_matrix = cosine_similarity(
            self.embeddings.cpu())  # Precompute the similarity matrix
        print('embeddings loaded:', emb_file)
        print('embeddings shape:', self.embeddings.shape)

        self.llm = model_dict['llm']
        self.prompt_dict = {}

        print('\nDONE! SAE initialized.')

    # get the top k nearest neighbors to the input sentence id
    # inputs: sentence_id (int), top_k (int)
    # outputs: top_neighbors (list)
    def get_top_neighbors(self, sentence_id, top_k=10):
        # get row from similarity matrix
        sim_row = self.embed_sim_matrix[sentence_id]
        # get the indices of the top k neighbors (excluding the input sentence)
        top_neighbors = np.argsort(-sim_row)[1:top_k+1]
        print('top neighbors:', top_neighbors)
        return top_neighbors

    # get the top k most similar features to the input feature
    # inputs: feature_id (int), k (int), top_percent (float)
    # outputs: sampled_features (list)
    def get_similar_features(self, feature_id, k=5, top_percent=0.001):
        sim_row = self.feature_sim_matrix[feature_id]
        sorted_row = np.argsort(-sim_row)[1:]
        # get top percent% of similar features
        top_k = int(len(sorted_row) * top_percent)
        top_features = sorted_row[1:top_k+1]  # exclude the input feature
        sampled_features = np.random.choice(top_features, k, replace=False)
        return sampled_features

    # add the input embedding to the embeddings and update the similarity matrix
    # inputs: emb (torch.Tensor)
    def add_embedding(self, emb):
        # if embeddings is an nxm matrix, emb should be a 1xm matrix
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)

        # Concatenate the new embedding
        self.embeddings = torch.cat((self.embeddings, emb.to(self.device)))

        # Compute similarities with all embeddings (including the new one)
        new_sim_matrix = cosine_similarity(self.embeddings.cpu())

        # Update the similarity matrix
        self.embed_sim_matrix = new_sim_matrix
        print('embedding added, new shape:', self.embeddings.shape)

    # remove the input embedding from the embeddings and update the similarity matrix
    # inputs: id (int)
    def remove_embedding(self, id):
        self.embeddings = torch.cat(
            (self.embeddings[:id], self.embeddings[id+1:]))
        new_sim_matrix = cosine_similarity(self.embeddings.cpu())
        self.embed_sim_matrix = new_sim_matrix
        print('embedding removed, new shape:', self.embeddings.shape)

    # get the existing embedding for the specified id
    # inputs: id (int)
    # outputs: sentence_embedding (torch.Tensor) or None
    def get_existing_embedding(self, id):
        if id >= 0 and id < len(self.embeddings):
            sentence_embedding = self.embeddings[id]
            return sentence_embedding
        return None

    # embed a single sentence with the specified embedding model
    # inputs: sentence (string)
    # outputs: embedding (torch.Tensor)
    def get_sentence_embedding(self, sentence):
        with torch.no_grad():
            device = self.device
            tk = self.tokenizer(
                [sentence],
                return_tensors="pt",
                padding=True,
                max_length=128,
                truncation=True,
            )

            result = self.embedding_model(
                input_ids=tk.input_ids.to(device),
                attention_mask=tk.attention_mask.to(device),
            )
            hidden_state = result.last_hidden_state
            emb = mean_pool(hidden_state, tk.attention_mask.to(device)).cpu()
            emb = emb.squeeze(0)
            return emb

    # add multiple sentence embeddings to the embeddings
    # inputs: sentences (list)
    # outputs: emb_list (list)
    def add_sentence_embeddings(self, sentences):
        # store all embeddings in a tensor
        emb_list = []
        for sentence in sentences:
            emb = self.get_sentence_embedding(sentence)
            emb_list.append(emb)
            self.add_embedding(emb)
        print('all embeddings added')
        # convert to tensor
        emb_list = torch.stack(emb_list)
        return emb_list

    # get the top k most similar features to the input embedding
    # inputs: embedding (torch.Tensor), top_k (int)
    # outputs: top_activations (list), similar_features (list)
    def get_top_activations(self, embedding, top_k=10):
        with torch.no_grad():
            device = self.device
            sae_inputs = embedding.to(device).to(torch.float32)
            feature_activations: np.ndarray = (
                self.sae.encode(sae_inputs).squeeze(0).cpu().numpy()
            )

            # Get the indices of the top k features
            all_features = np.argsort(-feature_activations)
            top_features = all_features[:top_k]

            # Get similar features for each of the top k features
            similar_features = []
            for feature in top_features:
                similar_features.extend(
                    self.get_similar_features(feature, k=5)
                )

            # Remove duplicates
            similar_features = np.unique(similar_features)
            # Exclude features that are already in top_features
            similar_features = np.setdiff1d(
                similar_features, top_features, assume_unique=True)

            # Create dictionary of top activations
            top_activations = {
                int(feature): float(feature_activations[feature])
                for feature in all_features
            }

            return top_activations, similar_features.tolist()

    # get the top k most similar features to the input sentence
    # inputs: sentence (string), id (int), top_k (int)
    # outputs: top_activations (list), similar_features (list)
    def get_top_activations_from_sentence(self, sentence, id, top_k=10):
        emb = self.get_existing_embedding(id)
        if emb is None:
            emb = self.get_sentence_embedding(sentence)
        print('finished embedding sentence')

        top_features, similar_features = self.get_top_activations(emb, top_k)
        # print('top_features:', top_features)
        print('got top activations')

        # Get the feature info for the top activations using the feature_info dataframe
        df_features = self.feature_info
        # 1) find the corresponding rows for the top features
        rows = df_features[df_features['feature'].isin(
            top_features.keys())].copy()
        # 2) add the activations to the rows
        rows['activation'] = rows['feature'].apply(
            lambda x: top_features[x])
        # 3) sort rows in rows by top_activations
        rows = rows.sort_values('activation', ascending=False)
        # 4) only save the feature, summary, activation columns
        rows = rows[['feature', 'summary', 'activation']]

        return rows, similar_features

    # for the given sentence, add the feature vectors and invert the
    # resultant embeddings to generate new sentences
    # inputs: sentence (string), id (int), features_and_weights (list)
    # outputs: new_sentences (list)
    def add_features_and_invert(self, sentence, id, features_and_weights):
        # features_and_weights is a list of dicts: {id: int, weight: float}

        # find the features for the given feature_ids + multiply by the weights
        feature_vectors = [self.features[feature['id']] * feature['weight']
                           for feature in features_and_weights]

        # add all the feature vectors to the sentence embedding
        sentence_embedding = self.get_existing_embedding(id)
        if sentence_embedding is None:
            sentence_embedding = self.get_sentence_embedding(sentence)
        new_embedding = sentence_embedding.cpu() + sum(feature_vectors)
        # convert to tensor
        new_embedding = new_embedding.to(self.device)
        # normalize the new embedding
        new_embedding = new_embedding / torch.norm(new_embedding)
        # invert the new embedding
        new_sentence = vec2text.invert_embeddings(
            new_embedding.unsqueeze(0), self.corrector)
        return new_sentence

    # project new embeddings to the UMAP space
    # inputs: new_embeddings (list)
    # outputs: new_umap_points (np.ndarray)
    def project_new_points(self, new_embeddings):
        # convert list of tensors to numpy
        new_embeddings = new_embeddings.cpu().numpy()
        # convert to 2D array
        # check if the new_embeddings are 2D
        if new_embeddings.ndim == 1:
            new_embeddings = new_embeddings.reshape(
                new_embeddings.shape[0], -1)
        new_umap_points = self.umap_reducer.transform(new_embeddings)
        return new_umap_points

    # reembed all sentences with UMAP and return the new points
    # outputs: new_points (list)
    def reembed_all_sentences(self):
        # reembed all sentences with new UMAP
        all_embeddings = self.embeddings
        new_umap = umap.UMAP(n_neighbors=100, min_dist=0.1,
                             n_components=2, metric='cosine')
        new_reducer = new_umap.fit(all_embeddings.cpu())
        new_umap_points = new_reducer.transform(all_embeddings.cpu())

        self.umap_reducer = new_reducer  # update the umap reducer

        # format the sentences and umap points to return as a list of dict objects
        new_points = format_new_points_umap(
            new_umap_points)
        return new_points

    # embed the new sentences, project the new embeddings to the UMAP space, and
    # format the new sentences and umap points to return as a list of dict objects
    # inputs: sentences (list)
    # outputs: new_points (list)
    def embed_and_format_points(self, all_sentences, gen_num, type='generate', weights=[]):
        # cut off the extra sentences if there are more than gen_num
        sentences = all_sentences[:gen_num]

        print('all sentences:')
        for s in sentences:
            print(s)
        # embed the new sentences
        all_embeddings = self.add_sentence_embeddings(sentences)
        # project the new embeddings to the UMAP space
        new_umap_points = self.project_new_points(all_embeddings)
        # format the new sentences and umap points to return as a list of dict objects
        if type == 'generate':
            new_points = format_new_points(sentences, new_umap_points)
        else:
            new_points = format_new_points_interpolate(
                sentences, new_umap_points, weights)
        return new_points

    # add the top features to the sentence and invert the embeddings
    # format the new sentences and umap points to return as a list of dict objects
    # inputs: sentence (string), id (number), features_and_weights (list), gen_num (int)
    # outputs: new_points (list)
    def generate_new_points(self, sentence, id, features_and_weights, gen_num):
        # add the features to the sentence and invert the embeddings
        new_sentence = self.add_features_and_invert(
            sentence, id, features_and_weights)
        # correct the new sentence using the llm model
        correct_start_time = time.time()
        corrected_sentence = correct_sentence(self.llm, sentence, new_sentence)
        correct_end_time = time.time()
        print('Sentence corrected in:', correct_end_time - correct_start_time)

        # generate variations of the corrected sentence
        extra_sentences_to_generate = gen_num - 1
        generate_start_time = time.time()
        sentence_variations = generate_sentence_variations(
            self.llm, extra_sentences_to_generate, corrected_sentence)
        generate_end_time = time.time()
        print(f'{extra_sentences_to_generate} variations generated in:',
              generate_end_time - generate_start_time)

        # embed and format the new sentences
        all_sentences = [corrected_sentence] + sentence_variations
        new_points = self.embed_and_format_points(all_sentences, gen_num)

        return new_points

    # generate new points using the language model
    # inputs: sentence (string), gen_num (int), instruction (string)
    # outputs: new_points (list)
    def generate_new_points_llm(self, sentence, gen_num, instruction):
        generate_start_time = time.time()
        new_sentences = prompt_for_sentence_variations_llm(
            self.llm, sentence, gen_num, instruction)
        generate_end_time = time.time()
        print(f'{gen_num} variations generated in:',
              generate_end_time - generate_start_time)

        # embed and format the new sentences
        new_points = self.embed_and_format_points(new_sentences, gen_num)

        return new_points

    # generate new points by interpolating between two sentences
    # inputs: sent1 (string), id1 (int), sent2 (string), id2 (int), gen_num (int)
    # outputs: new_points (list)
    def interpolate_between_points(self, sent1, id1, sent2, id2, gen_num):
        # get the embeddings for the two sentences
        emb1 = self.get_existing_embedding(id1)
        if emb1 is None:
            emb1 = self.get_sentence_embedding(sent1)
        emb2 = self.get_existing_embedding(id2)
        if emb2 is None:
            emb2 = self.get_sentence_embedding(sent2)

        # Ensure embeddings are of float type
        emb1 = emb1.to(torch.float32).to(self.device)
        emb2 = emb2.to(torch.float32).to(self.device)

        increment = 1 / (gen_num + 1)

        new_embeddings = []
        weights = np.arange(increment, 1.0, increment)
        for alpha in weights:
            # interpolate between the two embeddings
            mixed_embedding = torch.lerp(input=emb1, end=emb2, weight=alpha)
            # normalize the new embedding
            mixed_embedding = mixed_embedding / torch.norm(mixed_embedding)
            new_embeddings.append(mixed_embedding)

        new_embeddings = torch.stack(new_embeddings)  # convert to tensor

        # now invert all the new embeddings
        time_start = time.time()
        new_sentences = vec2text.invert_embeddings(
            new_embeddings, self.corrector)
        time_end = time.time()

        print(f'Interpolated {gen_num} sentences in:', time_end - time_start)

        # correct the new sentences using the llm model
        time_start = time.time()
        corrected_sentences = correct_multiple_sentences(
            self.llm, new_sentences, sent1, sent2)
        time_end = time.time()

        print(f'Sentences corrected in:', time_end - time_start)

        # embed and format the new sentences
        new_points = self.embed_and_format_points(
            corrected_sentences, gen_num, 'interpolate', weights)
        return new_points

    # add new sentence manually to dataset
    # inputs: sentence (string)
    # outputs: new_points (list)
    def add_new_sentence(self, sentence):
        new_points = self.embed_and_format_points([sentence], 1)
        return new_points

    # edit sentence in dataset
    # inputs: id (int), new_sentence (string)
    # outputs: new_points (list)
    def edit_sentence(self, id, new_sentence):
        # remove the old embedding
        new_embedding = self.get_sentence_embedding(new_sentence)
        # replace embedding at id with new_embedding
        self.embeddings[id] = new_embedding
        # recompute the similarity matrix
        new_sim_matrix = cosine_similarity(self.embeddings.cpu())
        self.embed_sim_matrix = new_sim_matrix
        # project the new embeddings to the UMAP space
        umap_points = self.project_new_points(new_embedding)
        # format the new sentences and umap points to return as a list of dict objects
        new_points = format_new_points([new_sentence], umap_points)
        return new_points

    # add prompt ideas to the dictionary for the given sentence
    # inputs: sentence (string), prompts (list)

    def add_prompts_to_dict(self, sentence, prompts):
        self.prompt_dict[sentence] = prompts
        print(f'Prompt ideas added to the dictionary for: {sentence}')

    # get prompt ideas for the given sentence
    # inputs: sentence (string), gen_num (int)
    # outputs: prompts (list)
    def get_prompt_ideas(self, sentence, gen_num=5):
        if sentence in self.prompt_dict:
            prompts = self.prompt_dict[sentence]
            print('Prompt ideas already generated:')
            for prompt in prompts:
                print(prompt)
            return prompts

        # else, generate prompt ideas for the given sentence
        start_time = time.time()
        prompts = generate_prompt_ideas(self.llm, gen_num, sentence)
        end_time = time.time()
        print(f'{gen_num} prompt ideas generated in:', end_time - start_time)
        for prompt in prompts:
            print(prompt)
        self.add_prompts_to_dict(sentence, prompts)
        return prompts
