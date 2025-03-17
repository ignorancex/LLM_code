import os
import pandas as pd
import torch

import os,sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from utils_image import load_pretrained_backbone_models_rl, load_pretrained_backbone_models
from parse_args_image import one, two, three
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DistilBertTokenizer
from transformers import DistilBertModel
from transformers import DistilBertPreTrainedModel
import numpy as np
# from models.Qmod import QNet
# from nlp_data_utils.nlp_dataset import *
from transformers import BertTokenizer
from transformers import BertModel

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, roc_auc_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.feature_extraction.text import TfidfTransformer
import nltk
from nltk import pos_tag
from nltk.corpus import wordnet
BERT_PRETRAINED_MODEL = 'bert-base-cased'
import pickle
from tqdm import tqdm

MASK_IDX=103
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
MASK_TOKEN = "[MASK]"
PAD_TOKEN_IDX = 0


model_name_str="name"
model_out_count_str="out_count"

model_property_mappings=[{model_name_str: "SamLowe/roberta-base-go_emotions", model_out_count_str:28}]


import openai

def extract_cacndidate_concepts(text):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    flag = True
    while(flag):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                    "role": "system",
                    "content": "You are an expert in identifying entities in a sentence. Please give me all the entities for a given sentence and separate them with commas. Below is an example:\n\nQ: \"I'm only giving this a neutral rating because it smelled lovely but arrived crushed, leaking and unable to be used. Very nice light fragrance - non irritating which is unusual as I tend to have a lot of allergies. This is not overpowering. Has an almost romantic and nostalgic fragrance.\"\n\nA: Neutral rating, Lovely smell, Crushed, Leaking, Very nice light fragrance, Non irritating, Allergies, Overpowering, Romantic and nostalgic fragrance"
                    },
                    {
                    "role": "assistant",
                    "content": text
                    }
                ],
                temperature=1,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
                )
            flag = False
        except openai.error.OpenAIError as e:
            print("Some error happened here.")
    
    
    
    concepts = response.choices[0]["message"]["content"].split(",")
    concepts = [concept.strip() for concept in concepts]
    return concepts

def get_embedding(concept_words):
	# Embed a line of text
	response = openai.Embedding.create(
    	model= "text-embedding-ada-002",
    	input=concept_words
	)
	# Extract the AI output embedding as a list of floats
	embedding = response["data"][0]["embedding"]
    
	return embedding

def get_embeddings_all(concept_words_ls):
    embedding_mappings = dict()
    for concept_words in tqdm(concept_words_ls):
        embedding = get_embedding(concept_words)
        embedding_mappings[concept_words] = embedding   
    return embedding_mappings

def populate_text_attrs(text):
    logits_ls = []    
    for model_idx in range(len(model_property_mappings)):
        model_name = model_property_mappings[model_idx][model_name_str]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, max_length=1024)
        inputs = tokenizer(text, return_tensors="pt",  max_length=1024, truncation=True)
        with torch.no_grad():
            inputs["input_ids"] = inputs["input_ids"][:,0:512]
            inputs["attention_mask"] = inputs["attention_mask"][:,0:512]
            logits = model(**inputs).logits
            logits_ls.append(logits)
    
    return torch.cat(logits_ls).numpy()

def obtain_pretrained_model_eeec(bert_state_dict=None):    
    if bert_state_dict:
        fine_tuned_state_dict = torch.load(bert_state_dict)
        bert = BertModel.from_pretrained(BERT_PRETRAINED_MODEL, state_dict=fine_tuned_state_dict)
    else:
        bert = BertModel.from_pretrained(BERT_PRETRAINED_MODEL)
    for p in bert.parameters():
        p.requires_grad = False
    return bert
        

def obtain_bow_features(text_ls, df, outcome_attr, classification, ngram=1, max_features=500):
    
    count_vectorizer = CountVectorizer()
    X_counts = count_vectorizer.fit_transform(text_ls)
    
    # tf_transformer = TfidfTransformer(use_idf=False).fit(X_counts)
    feature_names = count_vectorizer.get_feature_names_out()
    
   
    
    # text_feats = tf_transformer.transform(X_counts)
    text_feats = X_counts
    word_tfidf_sums = X_counts.sum(axis=0)
    word_tfidf_dict = {word: tfidf_sum for word, tfidf_sum in zip(feature_names, word_tfidf_sums.tolist()[0])}
    most_frequent_words_with_freq = sorted(word_tfidf_dict.items(), key=lambda x: x[1], reverse=True)
    
    most_frequent_words = [word for (word, _) in most_frequent_words_with_freq]
    
    nltk.download('averaged_perceptron_tagger')
    tags = pos_tag(most_frequent_words)
    
    def is_noun_or_verb(tag):
        allowed_tags = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS']
        return tag in allowed_tags
        # return tag.startswith('N') or tag.startswith('V') or tag.startswith('J')
    
    # filtered_words = [word for (word, _) in tags if is_noun_or_verb(wordnet.synset(word + '.n.01').pos()) or
    #               is_noun_or_verb(wordnet.synset(word + '.v.01').pos())]
    filtered_words = [word for (word, tag) in tags if is_noun_or_verb(tag)]
    
    most_frequent_words = filtered_words[:max_features]
    most_frequent_words_idx = [count_vectorizer.vocabulary_[word] for word in most_frequent_words]
    
    X_bow = text_feats[:, most_frequent_words_idx].toarray()
    
    outcome_labels = df[outcome_attr].to_numpy()
    if not classification:
        reg = LinearRegression().fit(X_bow, outcome_labels)
        pred_outcome_labels = reg.predict(X_bow)
        print("outcome l2 norm::", np.linalg.norm(outcome_labels.reshape(-1) - pred_outcome_labels.reshape(-1)))
    else:
        reg = LogisticRegression().fit(X_bow, outcome_labels)
        pred_outcome_labels = reg.predict(X_bow)
        pred_outcome_probs = reg.predict_proba(X_bow)
        print("outcome accuracy::", np.sum(outcome_labels.reshape(-1) == pred_outcome_labels.reshape(-1))/len(outcome_labels))
        print("outcome loss::", torch.nn.functional.cross_entropy(torch.tensor(pred_outcome_probs), torch.tensor(outcome_labels)))
    
    # word_ls = tf_transformer.get_feature_names_out()
    # word_ls = [word_ls[idx] for idx in range(max_features)]
    
    df[most_frequent_words] = pd.DataFrame(X_bow)
    
    return df



def extract_candidate_concepts_all(text_ls):
    all_concepts = []
    all_concept_ls = []
    for text in tqdm(text_ls):
        # sub_text_ls = text.strip().split(". ")
        # for sub_text in sub_text_ls:
        #     sub_sub_text_ls = sub_text.split("\n")
        #     for sub_sub_text in sub_sub_text_ls:
            curr_concept_ls = extract_cacndidate_concepts(text.strip())
            all_concepts.extend(curr_concept_ls)
            all_concept_ls.append(curr_concept_ls)
    return all_concepts, all_concept_ls

def cluster_concept_embeddings(embedding_ls, cluster_count = 50):
    embedding_array = np.array(embedding_ls)
    kmeans = KMeans(n_clusters=cluster_count, random_state=0).fit(embedding_array)
    assigned_cluster_ids = kmeans.predict(embedding_array)
    
    dist_to_clusters = np.linalg.norm(embedding_array.reshape(len(embedding_array), 1, -1) - kmeans.cluster_centers_.reshape(1, len(kmeans.cluster_centers_), -1), axis=2)
    
    closet_samples=np.argmin(dist_to_clusters, axis=0)
    
    return assigned_cluster_ids, closet_samples


def k_means(data, k, max_iters=100):
    # Initialize centroids randomly from the data points
    centroids = data[torch.randperm(data.shape[0])[:k]]
    
    for iter in tqdm(range(max_iters)):
        # Calculate distances from data points to centroids
        distances = torch.cdist(data, centroids)
        
        # Assign each data point to the closest centroid
        _, cluster_assignments = distances.min(dim=1)
        
        # Update centroids based on the mean of data points in each cluster
        new_centroids = torch.stack([data[cluster_assignments == i].mean(dim=0) if torch.sum(cluster_assignments == i) > 0 else centroids[i] for i in range(k)])
        
        # Check for convergence
        if torch.all(new_centroids == centroids):
            break
        
        centroids = new_centroids
    
    return centroids, cluster_assignments

# def obtain_concept_features(log_path, text_ls, df, cluster_count=50):
    
#     cached_concept_file = os.path.join(log_path, "all_concepts")
#     cached_concept_ls_f

#     all_concepts, all_concept_ls = extract_candidate_concepts_all(text_ls)
#     with open(os.path.join(log_path, "all_concepts"), "wb") as f:
#         pickle.dump(all_concepts, f)
#     with open(os.path.join(log_path, "all_concepts_ls"), "wb") as f:
#         pickle.dump(all_concept_ls, f)
#     word_to_embedding_mappings = get_embeddings_all(all_concepts)
    
#     with open(os.path.join(log_path, "word_to_embedding_mappings"), "wb") as f:
#         pickle.dump(word_to_embedding_mappings, f)
    
#     embedding_ls = [word_to_embedding_mappings[concept] for concept in all_concepts]
#     assigned_cluster_ids, closet_sample_ids = cluster_concept_embeddings(embedding_ls, cluster_count=cluster_count)
#     word_to_cluster_id_mappings = {concept: cluster_id for concept, cluster_id in zip(all_concepts, assigned_cluster_ids)}
#     all_features = np.zeros((len(text_ls), cluster_count))
#     for text_id in range(len(text_ls)):
#         curr_concept_ls = all_concept_ls[text_id]
#         for concept in curr_concept_ls:
#             all_features[text_id, word_to_cluster_id_mappings[concept]] += 1
    
#     concept_name_ls = [all_concepts[k] for k in closet_sample_ids]
#     df[concept_name_ls] = all_features
    
#     return all_features
    
    
    # for sub_concept_ls in all_concept_ls:
        
    
    
    



def populate_text_attrs_all(all_text, df):
    attr_name_ls = []
    for model_idx in range(len(model_property_mappings)):
        attr_name_ls.extend([model_property_mappings[model_idx][model_name_str] + "_" + str(k) for k in range(model_property_mappings[model_idx][model_out_count_str])])
    
    all_text_ls = list(all_text)
    for idx in tqdm(range(len(all_text_ls))):
        text = df.loc[idx, "text"]# all_text_ls[idx]
        curr_attr_values = populate_text_attrs(text)
        df.loc[idx, attr_name_ls] = curr_attr_values.reshape(-1)
    
    return df

def populate_text_attrs_all_2(all_text, df, ngram=1, max_features=500):
    ngram_vectorizer = CountVectorizer(ngram_range=(ngram,ngram),
                                        stop_words='english', max_features=max_features)
    vectorized_data = ngram_vectorizer.fit_transform(all_text).toarray()
    
    feat_name_ls = ["ngram_" + str(k) for k in range(max_features)]
    
    # df[feat_name_ls] = pd.DataFrame(vectorized_data)
    for idx in tqdm(range(len(vectorized_data))):
        text = df.loc[idx, "text"]# all_text_ls[idx]
        assert text == all_text[idx]
        df.loc[idx, feat_name_ls] = vectorized_data[idx].tolist()
    return df

def truncate_seq_first(tokens, max_seq_length):
    max_num_tokens = max_seq_length - 2
    trunc_tokens = list(tokens)
    if len(trunc_tokens) > max_num_tokens:
        trunc_tokens = trunc_tokens[:max_num_tokens]
    return trunc_tokens

def transform_text_to_tokens(tokenizer, text, dataset_name):
    if dataset_name == "music":
        max_length = 128
        encoded_sent = tokenizer.encode_plus(text, add_special_tokens=True,
                                                max_length=max_length,
                                                truncation=True,
                                                pad_to_max_length=True)
        text_ids, text_mask, text_len = torch.tensor(encoded_sent['input_ids']), torch.tensor(encoded_sent['attention_mask']), torch.tensor(sum(encoded_sent['attention_mask']))
        return text_ids, text_mask, text_len
        
    elif dataset_name == "EEEC":
        max_seq_length = 32
        tokens = tokenizer.tokenize(text)

        tokens = tuple([CLS_TOKEN] + truncate_seq_first(tokens, max_seq_length) + [SEP_TOKEN])

        example_len = len(tokens) - 2

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(PAD_TOKEN_IDX)
            input_mask.append(PAD_TOKEN_IDX)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        
        return torch.tensor(input_ids), torch.tensor(input_mask), torch.tensor(example_len)


def populate_text_attrs_all_3(dataset_name, all_text, df, treatment_attr, outcome_attr, text_attr, tokenizer, bert_model, device, max_features=50):

    

    all_pooled_output_ls = []

    with torch.no_grad():
        text_ids_ls = []
        text_mask_ls = []
        text_len_ls= []
        for text in tqdm(list(all_text)):
            text_ids, text_mask, text_len = transform_text_to_tokens(tokenizer, text, dataset_name)
            text_ids_ls.append(text_ids)
            text_mask_ls.append(text_mask)
            text_len_ls.append(text_len)
        mb_size = 64
        for start_id in tqdm(range(0, len(text_ids_ls), mb_size)):
            end_id = start_id + mb_size
            if end_id >= len(text_ids_ls):
                end_id = len(text_ids_ls)
            text_ids = torch.stack(text_ids_ls[start_id:end_id]).to(device)
            text_mask = torch.stack(text_mask_ls[start_id:end_id]).to(device)
            text_len = torch.stack(text_len_ls[start_id:end_id]).to(device)
            
            text_len = text_len.unsqueeze(1) - 2  # -2 because of the +1 below
            attention_mask_class = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
            mask = (attention_mask_class(text_len.shape).uniform_() * text_len.float()).long() + 1  # + 1 to avoid CLS
            target_words = torch.gather(text_ids, 1, mask)
            mlm_labels = torch.ones(text_ids.shape).long() * -100
            if torch.cuda.is_available():
                mlm_labels = mlm_labels.cuda()
            mlm_labels.scatter_(1, mask, target_words)
            text_ids.scatter_(1, mask, MASK_IDX)
            text_ids = text_ids.to(device)
            text_mask = text_mask.to(device)
            if dataset_name == "music":
                pooled_output = bert_model(input_ids=text_ids, attention_mask=text_mask)[0][:,0]
            else:
                pooled_output = bert_model(input_ids=text_ids, attention_mask=text_mask)[0][:,0]#.view(text_ids.shape[0], -1)
            all_pooled_output_ls.append(pooled_output.cpu())
    all_pooled_output_array = torch.cat(all_pooled_output_ls)
    # W, assigned, vectorized_data = k_means_pytorch(all_pooled_output_array, max_features)
    # s_score = silhouette_score(all_pooled_output_array.cpu().numpy(), assigned.cpu().numpy())
    # print("silhouette_score::", s_score)
    
    vectorized_data, reconstruction_error = pca_decomposition(all_pooled_output_array, max_features)
    labels = df[treatment_attr].to_numpy()
    
    clf = LogisticRegression(random_state=0).fit(vectorized_data.cpu().numpy(), labels)
    pred_labels = clf.predict_proba(vectorized_data.cpu().numpy())
    auc_score = roc_auc_score(labels, pred_labels[:,1])
    
    print("treatment auc score::", auc_score)
    
    outcome_labels = df[outcome_attr].to_numpy()
    reg = LinearRegression().fit(vectorized_data.cpu().numpy(), outcome_labels)
    pred_outcome_labels = reg.predict(vectorized_data.cpu().numpy())
    
    print("outcome l2 norm::", np.linalg.norm(outcome_labels.reshape(-1) - pred_outcome_labels.reshape(-1)))
    
    
    
    print("reconstruction errors::", reconstruction_error)
    print("full data norm::", torch.norm(all_pooled_output_array))
    print("reconstructed data norm::", reconstruction_error/torch.norm(all_pooled_output_array))
    
    # s_score = calculate_silhouette_score(all_pooled_output_array, assigned)
    
    # reducer = KMeans(n_clusters=max_features).fit(all_pooled_output_array)
    # W = reducer.cluster_centers_.astype(np.float32)
    # vectorized_data = reducer.transform(all_pooled_output_array)
    
    feat_name_ls = ["dist_" + str(k) for k in range(max_features)]
    
    # df[feat_name_ls] = pd.DataFrame(vectorized_data)
    for idx in tqdm(range(len(vectorized_data))):
        text = df.iloc[idx][text_attr]# all_text_ls[idx]
        assert text == all_text[idx]
        df.loc[idx, feat_name_ls] = vectorized_data[idx].tolist()
    return df

def pca_decomposition(matrix, num_components):
    # Center the data by subtracting the mean along each feature
    mean = torch.mean(matrix, dim=0)
    centered_data = matrix - mean
    
    # Perform singular value decomposition (SVD)
    U, S, V = torch.svd(centered_data)
    
    # Select the top num_components principal components (eigenvectors)
    top_components = V[:, :num_components]
    
    # Project the data onto the lower-dimensional subspace defined by the principal components
    pca_result = torch.mm(centered_data, top_components)
    
    reconstructed_data = torch.mm(pca_result, top_components.t()) + mean
    
    # Compute the reconstruction error
    reconstruction_error = torch.norm(matrix - reconstructed_data)
    
    return pca_result, reconstruction_error