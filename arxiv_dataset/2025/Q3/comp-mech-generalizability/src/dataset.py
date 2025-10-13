from abc import abstractmethod
import random
import os
import torch
import gensim.downloader as api
from torch import Tensor
from torch.utils.data import Dataset
import json
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Literal, Union

from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM

from model import BaseModel, WrapHookedTransformer
import numpy as np
import pandas as pd
from line_profiler import profile

LOG_DIR = "../logs"
REDC = "\033[91m"
ENDC = "\033[0m"


DOMAINS = ['Adult', 'Arts_and_Entertainment', 'Autos_and_Vehicles', 'Beauty_and_Fitness',
           'Books_and_Literature', 'Business_and_Industrial', 'Computers_and_Electronics', 'Finance',
           'Food_and_Drink', 'Games', 'Health', 'Hobbies_and_Leisure', 'Home_and_Garden',
           'Internet_and_Telecom', 'Jobs_and_Education', 'Law_and_Government', 'News',
           'Online_Communities', 'People_and_Society', 'Pets_and_Animals', 'Real_Estate', 'Science',
           'Sensitive_Subjects', 'Shopping', 'Sports', 'Travel_and_Transportation']

def load_dataset(
    path: str, experiment: str,
        start: Optional[int],
        end: Optional[int],
        domain=None
) -> List[Dict]:
    data = json.load(open(path, "r"))
    if start is None:
        start = 0
    if end is None:
        end = len(data)
    if domain and isinstance(domain, str) and experiment == "copyVSfactDomain":
        print(f"Filtering for Domain {domain}!")
        data = [row for row in data if row["domain"] == domain]
    return data[start:end]


def load_similarity_score_dict(model_name: str) -> Optional[Dict]:
    # family_name = get_family_name(model_name)
    # if os.path.exists(f"../data/similarity_score_{family_name}.pt"):
    #     return torch.load(f"../data/similarity_score_{family_name}.pt")
    # else:
    #     return None
    if os.path.exists(f"../data/similarity_score_all_dict.pt"):
        return torch.load(f"../data/similarity_score_all_dict.pt")
    else:
        return None
    
def load_word_embeddings_dict(model_name: str) -> Optional[Dict]:
    if os.path.exists(f"../data/word_embeddings_word2vec.pt"):
        return torch.load(f"../data/word_embeddings_word2vec.pt")
    else:
        return None


def get_family_name(model_name: str) -> str:
    if "gpt2" in model_name:
        return "gpt2"
    elif "pythia" in model_name:
        return "pythia"
    else:
        raise NotImplementedError(f"Model {model_name} is not supported")


def compute_similarity_for_target(args):
    word2vec, d, similarity_score_dict = args
    target_word = d["target_true"].strip()

    similarity = []
    for word in word2vec.vocab:
        vector = word2vec[word]
        sim = np.dot(vector, word2vec[target_word]) / (np.linalg.norm(vector) * np.linalg.norm(word2vec[target_word]))
        similarity.append((word, sim))

    similarity_score_dict[target_word] = similarity

class BaseDataset(Dataset):
    def __init__(
        self,
        path: str,
        model: BaseModel,
        experiment: Literal["copyVSfact", "contextVSfact"],
        start: Optional[int] = None,
        end: Optional[int] = None,
        similarity: Tuple[
            bool, int, Literal["self-similarity", "modify-self-similarity", "data-sampling"]
        ] = (False, 0, "self-similarity"),
        premise: str = "Redefine",
        no_subject: bool = False,
        prompt_type: str = None,
        domain: str = None
    ):
        if no_subject:
            print(
                f"{REDC} No subject found in the dataset {ENDC}, proceeding with no subject data"
            )
        self.no_subject = no_subject
        self.model = model
        self.experiment = experiment
        self.similarity = similarity
        self.premise = premise
        self.prompt_type = prompt_type
        self.domain = domain
        if similarity[0]:
            self.full_data = load_dataset(path, self.experiment, start, end, self.domain)
            self.similarity_score_dict = load_similarity_score_dict(
                self.model.cfg.model_name
            )
            self.word_embeddings_dict = load_word_embeddings_dict(
                self.model.cfg.model_name
            )
            self.full_data = self.generate_similarity_data(similarity[2])
            self.base_full_data = self.full_data
        else:
            self.full_data = load_dataset(path, self.experiment, start, end, self.domain)

        self.lengths = self.__get_lenghts_and_tokenize__()
        if similarity[0]:
            self.filtered_data = self.filter_similarity_data()
        self.prompts = []
        self.tokenized_prompts = []
        self.targets = []
        self.obj_pos = []
        self.first_subj_pos = []
        self.second_subj_pos = []
        self.subj_len = []

    def reset(
        self,
        new_similarity_level: Optional[int] = None,
    ):
        if self.similarity[0] is True:
            self.similarity = (
                self.similarity[0],
                new_similarity_level,
                self.similarity[2],
            )

        self.lengths = self.__get_lenghts_and_tokenize__()
        if self.similarity[0]:
            self.filtered_data = self.filter_similarity_data()
        self.prompts = []
        self.tokenized_prompts = []
        self.targets = []
        self.obj_pos = []
        self.first_subj_pos = []
        self.second_subj_pos = []

        self.subj_len = []

    def update(
        self,
        premise: str,
        new_similarity_level: Optional[int] = None,
    ):
        print(
            f"Updating the dataset from {self.premise} to {premise} and the similarity level from {self.similarity[1]} to {new_similarity_level}"
        )
        self.similarity = (self.similarity[0], new_similarity_level, self.similarity[2])
        self.premise = premise
        self.prompts = []
        self.tokenized_prompts = []
        self.targets = []
        self.obj_pos = []
        self.lengths = []
        self.subj_len = []
        self.lengths = self.__get_lenghts_and_tokenize__()
        if self.similarity[0]:
            self.filtered_data = self.filter_similarity_data()

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        if len(self.prompts) == 0:
            raise ValueError("Dataset is empty: please call set_len() first")
        else:
            return {
                "prompt": self.prompts[idx],
                "input_ids": self.tokenized_prompts[idx],
                "target": self.targets[idx],
                "obj_pos": self.obj_pos[idx],
                "1_subj_pos": self.first_subj_pos[idx],
                "2_subj_pos": self.second_subj_pos[idx],
                "subj_len": self.subj_len[idx],
            }

    def get_lengths(self):
        return self.lengths

    def modify_prompt(self, prompt_type, row):
        ## Fewshot Prompt
        # one_shot_example = "Question: iPhone X was developed by Google? Answer: False\n"
        # one_shot_example += "Question: Samsung S10 was developed by Samsung? Answer: True\n"

        ## Premise change
        # "prompt" = row["prompt"].replace("Redefine: ", "Fact Check: ")
        # "prompt": row["prompt"].replace("Redefine: ", "Validate: ")
        # "prompt": row["prompt"].replace("Redefine: ", "Verify: ")

        ## Template change - True/False
        # "prompt": one_shot_example + "Question: " + row["base_prompt"] + f"{row['target_new']}? Answer:",
        # "target_true": "True",
        # "target_new": "False"

        ## Template change - Context Question
        if prompt_type == "context_qna":
            modified_prompt = "Context: " + row["base_prompt"] + f"{row['target_new']}. " + \
                              "Question: " + row["base_prompt"]

        ## Template change - Statement 1 (Fact) / 2 (Counterfact)
        elif prompt_type == "fact_check_v1":
            modified_prompt = "Statement 1: " + row["base_prompt"] + f"{row['target_true']}. " + \
                              "Statement 2: " + row["base_prompt"] + f"{row['target_new']}. " + \
                              "Fact Check: " + row["base_prompt"]
        ## Template change - Statement 1 (Counterfact) / 2 (Fact)
        elif prompt_type == "fact_check_v2":
            modified_prompt = "Statement 1: " + row["base_prompt"] + f"{row['target_new']}. " + \
                              "Statement 2: " + row["base_prompt"] + f"{row['target_true']}. " + \
                              "Fact Check: " + row["base_prompt"]
        else:
            modified_prompt = self.__get_prompt__(row)

        return modified_prompt

    def __get_prompt__(self, d: Dict) -> str:
        if self.experiment in ["copyVSfact", "copyVSfactDomain"]:
            return d["template"].format(self.premise, d["target_new"])
        elif self.experiment == "copyVSfactQnA":
            return d["prompt"]
        else:
            raise NotImplementedError(f"Experiment {self.experiment} is not supported")

    def __find_first_occurence__(
        self, prompt: torch.Tensor, target: torch.Tensor
    ) -> int:
        position = -1
        for i in range(prompt.shape[0] - target.shape[0] + 1):
            if torch.all(prompt[i : i + target.shape[0]] == target):
                position = i
                break
        return position

    def __find_second_occurence__(
        self, prompt: torch.Tensor, target: torch.Tensor, first_occurence: int
    ) -> int:
        position = -1
        for i in range(first_occurence + 1, prompt.shape[0] - target.shape[0] + 1):
            if torch.all(prompt[i : i + target.shape[0]] == target):
                position = i
                break
        return position

    def find_occurence(
        self, prompt: torch.Tensor, target: torch.Tensor
    ) -> Tuple[int, int]:
        first_occurence = self.__find_first_occurence__(prompt, target)
        second_occurence = self.__find_second_occurence__(
            prompt, target, first_occurence
        )
        return first_occurence, second_occurence

    def __find_subj_pos__(self, d: Dict) -> Tuple[int, int, int]:
        if self.no_subject:
            return -1, -1, -1
        subject_string = " " + d["subject"]
        subject_token = self.model.tokenize(subject_string).squeeze(0).cpu()
        prompt_token = d["tokenized_prompt"]

        subject_token_len = subject_token.shape[0]
        # find the first occurence of the subject tokens in the prompt tokens
        first_subj_pos, second_subj_pos = self.find_occurence(
            prompt_token, subject_token
        )

        if first_subj_pos == -1:
            # try with removing the last token
            subject_token = subject_token[:-1]
            subject_token_len = subject_token.shape[0]
            first_subj_pos, second_subj_pos = self.find_occurence(
                prompt_token, subject_token
            )

        if first_subj_pos == -1:
            raise ValueError(f"Subject token: {subject_token}")

        return first_subj_pos, second_subj_pos, subject_token_len - 1

    def __find_obj_pos__(self, d: Dict) -> int:
        object_string = d["target_new"]
        object_token = self.model.tokenize(object_string).cpu()
        prompt_token = d["tokenized_prompt"]

        # find the first occurence of the subject tokens in the prompt tokens
        obj_pos = -1
        for i in range(prompt_token.shape[0] - object_token.shape[0] + 1):
            if torch.all(prompt_token[i : i + object_token.shape[0]] == object_token):
                obj_pos = i
                break
        return obj_pos

    def one_token(self, token: torch.Tensor) -> torch.Tensor:
        if token.shape[0] == 1:
            return token
        else:
            return token[0].unsqueeze(0)

    def __get_lenghts_and_tokenize__(self):
        lengths = []
        log_data = []
        to_remove = []
        if self.similarity[0] and self.similarity[2] == "data-sampling":
            self.full_data = [d for d in self.base_full_data if d["similarity_group"] == self.similarity[1]]
           
        print(f"Example prompt: {self.full_data[0]}")
        for d in tqdm(self.full_data, desc="Tokenizing and computing lengths"):
            if self.similarity[0] and self.similarity[2] == "data-sampling":
                d["target_new"] = random.choice(d["target_new_list"])
            if self.prompt_type in ["qna"]:
                pass
            elif self.prompt_type in ["fact_check_v1", "fact_check_v2", "context_qna"]:
                d["prompt"] = self.modify_prompt(self.prompt_type, d)
            else:
                d["prompt"] = self.__get_prompt__(d)
            d["tokenized_prompt"] = self.model.tokenize(d["prompt"]).squeeze(0).cpu()
            d["target_new_token"] = self.one_token(
                self.model.tokenize(d["target_new"]).squeeze(0).cpu()
            )
            d["target_true_token"] = self.one_token(
                self.model.tokenize(d["target_true"]).squeeze(0).cpu()
            )
            d["targets"] = torch.cat(
                (d["target_true_token"], d["target_new_token"]), dim=0
            )
            # if find_subj_pod raises an error, add the point to the log data and continue
            try:
                first_subj_pos, second_subj_pos, subj_len = self.__find_subj_pos__(d)
                d["1_subj_pos"] = first_subj_pos
                d["2_subj_pos"] = second_subj_pos
                d["subj_len"] = subj_len

            except ValueError as e:
                for key in d.keys():
                    if isinstance(d[key], torch.Tensor):
                        d[key] = d[key].tolist()
                log_data.append((d, str(e)))
                # remove the point from the full data
                to_remove.append(d)
                continue

            d["obj_pos"] = self.__find_obj_pos__(d)
            d["length"] = d["tokenized_prompt"].shape[0]
            if d["length"] not in lengths:
                lengths.append(d["length"])
        print(f"Example prompt after processing: {self.full_data[0]}")
        if len(log_data) > 0:
            print(
                f"{REDC} Found {len(log_data)} errors while tokenizing the prompts. Check the logs for more details... {ENDC}"
            )
            if not os.path.exists(LOG_DIR):
                os.makedirs(LOG_DIR)

            # save in a json file
            with open(
                f"../logs/tokenization_errors_{self.model.cfg.model_name}.json", "w"
            ) as f:
                json.dump(log_data, f, indent=4)

        for d in to_remove:
            self.full_data.remove(d)

        return lengths

    def generate_similarity_data(
        self, similarity_type: Literal["data-sampling", "self-similarity", "modify-self-similarity"]
    ):
        if similarity_type == "self-similarity":
            return self.__generate_self_similarity_data__()
        elif similarity_type == "modify-self-similarity":
            return self.__generate_modify_self_similarity_data__()
        elif similarity_type == "data-sampling":
            return self.__generate_similarity_data_sampling_fast__()
        else:
            raise NotImplementedError(
                f"Similarity type {similarity_type} is not supported"
            )
            
    @profile
    def __generate_similarity_data_sampling_fast__(self):
        
        # load the model
        
        if self.word_embeddings_dict is None:
            word2vec = api.load("word2vec-google-news-300")
        # declare the embedding tensor and the mapping index
            word_embeddings = torch.zeros(len(word2vec.vocab), 300, dtype=torch.float16) # type: ignore
            # word_index = pd.DataFrame(columns=["word", "index"])
            word_index = {}
            for i, word in tqdm(enumerate(word2vec.vocab.keys()), desc="converting embedding to torch", total=len(word2vec.vocab)): #type: ignore
                word_index[word] = word2vec.vocab[word].index #type: ignore
                word_embeddings[i] = torch.tensor(word2vec[word], dtype=torch.float16)
                
                
            word_embeddings = word_embeddings.to("cuda")
            #release memory
            del word2vec
            
            word_index = pd.DataFrame(list(word_index.items()), columns=["word", "index"])
            word_index.set_index("word", inplace=True)
            print("Word2Vec released - Converted to Torch")
            torch.save({
                "word_embeddings": word_embeddings,
                "word_index": word_index
            }, "../data/word_embeddings_word2vec.pt")
        else:
            word_embeddings = self.word_embeddings_dict["word_embeddings"]
            word_index = self.word_embeddings_dict["word_index"]
        

        similarity_score_per_word = {}
        full_norm = torch.norm(word_embeddings, dim=1)
        for d in tqdm(
            self.full_data,
            desc="Computing all similarities (word2vec)",
            total=len(self.full_data),
        ):
            target_word = d["target_true"].lstrip()
            if target_word in similarity_score_per_word:
                continue
            
            target_vector = word_embeddings[word_index.loc[target_word,"index"]].cpu()
            
            #compute the similarity between target vector and all the other
            dot_products = torch.matmul(word_embeddings, target_vector.unsqueeze(1)).squeeze(1) # shape 
            
            cosine_similarities = dot_products /  (full_norm * torch.norm(target_vector))
            
            similarity_score_per_word[target_word] = cosine_similarities.cpu()
            
            
        similarity_score_dict = {
            "similarity_score_dict": similarity_score_per_word,
            "word_index": word_index,
        }
            
        torch.save(similarity_score_dict, "../data/similarity_score_all_dict.pt")
        self.similarity_score_dict = similarity_score_dict
            
    #     @profile
    #     def process_data_point(d):
    #         similarity_score_dict = self.similarity_score_dict["similarity_score_dict"]
    #         word_index = self.similarity_score_dict["word_index"]
    #         base_target = d["target_true"].lstrip()
    #         similarity_scores = similarity_score_dict[base_target].to(torch.float32)
    #         quantilies = torch.quantile(similarity_scores, torch.linspace(0,1,11))
    #         indices = torch.bucketize(similarity_scores, quantilies)
    #         word_index["group"] = indices
    #         # map the indices to the word
            
            
    #         local_new_data = []
    #         for grup in range(1, 11):
    #             new_d = d.copy()
                
    #             new_d["target_new_list"] = word_index[word_index["group"] == grup].sample(20).index.tolist()
    #             new_d["similarity_group"] = grup
    #             local_new_data.append(new_d)
    #         return local_new_data
        
    #     import concurrent.futures
        
    # # Use ThreadPoolExecutor with 4 workers and tqdm for progress tracking
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
    #         # Prepare a list to store future objects
    #         futures = [executor.submit(process_data_point, d) for d in self.full_data]
            
    #         # Iterate over futures and update tqdm progress bar
    #         results = []
    #         for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing Data Points"):
    #             results.append(future.result())
                
    #     new_data = []
    #     for result in results:
    #         new_data.extend(result)
                
    #     return new_data
        new_data = []
        similarity_score_dict = self.similarity_score_dict["similarity_score_dict"]
        word_index = self.similarity_score_dict["word_index"]
        word_index["word"] = word_index.index
        for d in tqdm(self.full_data, desc="Generating sampled-similarity tokens (word2vec)", total=len(self.full_data)):
            base_target = d["target_true"].lstrip()  # strip leading spaces efficiently

            similarity_scores = similarity_score_dict[base_target]
            
            # to dtype=torch.float32
            similarity_scores = similarity_scores.to(torch.float32).cpu()

            # Compute quantiles
            quantiles = torch.quantile(similarity_scores, torch.linspace(0, 1, 11).cpu())

            # Use vectorized operations to categorize words
            indices = torch.bucketize(similarity_scores, quantiles).cpu()  # This will assign each score to a bucket
            
            # map the indices to the words
            # word_groups = [word_index.iloc[indices == i].index.tolist() for i in range(10)]
            #assign each word to a group based on the indices
            #add indices to the word_index
            word_index["group"] = indices
            #add the current word_index as a column to the word_index
            
            # word_index = word_index.reset_index(drop=True)
            word_index = word_index.set_index("group")

                
            
            for grup in range(1,11):
                #for each group create a new data point
                new_d = d.copy()
                # select 20 words from the group
                # new_d["target_new"] = [ random.choice(grup) for _ in range(20)]
                # sample 20 words from word_index based on the group
                # new_d["target_new_list"] = word_index[word_index["group"] == grup].sample(200).index.tolist()
                new_d["target_new_list"] = word_index.loc[grup].sample(20).word.tolist()
                new_d["similarity_group"] = grup
                new_data.append(new_d)
                
        return new_data
            

        
        
            
        
            
    def __generate_similarity_data_sampling__(self):
        word2vec = api.load("word2vec-google-news-300")
        print("Word2Vec loaded")
        self.similarity_score_dict = None
        from multiprocessing import Pool, Manager
        from concurrent.futures import ProcessPoolExecutor
        import concurrent.futures
        
        def word_vector_batch_generator(word2vec, batch_size=100):
            """
            Generator that yields a batch of vectors as a tensor and corresponding words as a list.
            """
            vectors = []
            words = []
            for word in word2vec.vocab:
                vectors.append(word2vec[word])
                words.append(word)
                if len(vectors) == batch_size:
                    yield torch.tensor(np.array(vectors)), words  # Convert list of arrays to a single numpy array before tensor conversion
                    vectors = []
                    words = []
            if vectors:
                yield torch.tensor(np.array(vectors)), words  # Convert list of arrays to a single numpy array before tensor conversion

        def similarity_computation(word2vec, target_word, device):
            """
            Compute the cosine similarity of `target_word` against all words in `word2vec`.
            Returns a list of (word, similarity) tuples.
            """
            # target_vec = torch.tensor(word2vec[target_word]).to(device)
            target_vec = torch.tensor(word2vec[target_word])
            target_norm = torch.norm(target_vec)
            similarities = []

            for vectors, words in word_vector_batch_generator(word2vec):
                # vectors = vectors.to(device)
                norms = torch.norm(vectors, dim=1)

                # Broadcasting to compute dot product
                dot_products = torch.matmul(vectors, target_vec.unsqueeze(1)).squeeze(1)
                sims = dot_products / (norms * target_norm)

                similarities.extend(zip(words, sims.cpu().numpy()))

            return similarities
        

        if self.similarity_score_dict is None:
            # Manager for managing a shared dictionary
            similarity_score_dict = {}
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                futures = {}
                for d in tqdm(self.full_data, desc="Scheduling tasks"):
                    target_word = d["target_true"].strip()
                    if target_word not in similarity_score_dict:
                        future = executor.submit(similarity_computation, word2vec, target_word, device)
                        futures[future] = target_word

                for future in tqdm(concurrent.futures.as_completed(futures), desc="Calculating similarities", total=len(futures)):
                    target_word = futures[future]
                    similarity_score_dict[target_word] = future.result()
            self.similarity_score_dict = similarity_score_dict
            torch.save(similarity_score_dict, f"../data/similarity_score_{self.model.cfg.model_name}.pt")
            
    def __generate_similarity_data_sampling___(self):
        word2vec = api.load("word2vec-google-news-300")
        print("Word2Vec loaded")
        #if exist the similarity score dict, use it
        self.similarity_score_dict = None
        if self.similarity_score_dict is None:
            
            def word_vector_generator(word2vec):
                for word in word2vec.vocab:
                    yield word2vec[word], word
            
            
            similarity_score_dict = {}
            for d in tqdm(
                self.full_data,
                desc="Computing all similarities (word2vec)",
                total=len(self.full_data),
            ):
                vector_iterator = word_vector_generator(word2vec)
                target_word = d["target_true"]
                # remove first space if present
                if target_word[0] == " ":
                    target_word = target_word[1:]
                    
                # skip if the target word is already in the similarity score dict
                if target_word in similarity_score_dict:
                    continue
                

                
                similarity = []
                for vector, word in vector_iterator:
                    sim = np.dot(vector, word2vec[target_word]) / (np.linalg.norm(vector) * np.linalg.norm(word2vec[target_word]))
                    similarity.append((word, sim))

                similarity_score_dict[target_word] = similarity
            
            torch.save(similarity_score_dict, f"../data/similarity_score_{self.model.cfg.model_name}.pt")
            self.similarity_score_dict = similarity_score_dict
        
        new_data = []
        # for d in tqdm(
        #     self.full_data,
        #     desc="Generating sampled-similarity tokens (word2vec)",
        #     total=len(self.full_data),
        # ):
        #     base_target = d["target_true"]
            
        #     if base_target[0] == " ":
        #         base_target = base_target[1:]
                
        #     #get a tensor of similarity scores for the target word
        #     similarity_scores = torch.tensor([sim for word, sim in self.similarity_score_dict[base_target]])
        #     # word_list = [word for word, sim in self.similarity_score_dict[base_target]]
            
        #     # divide in 10 groups based on the quantiles
        #     groups = torch.quantile(similarity_scores, torch.linspace(0, 1, 11))
            
        #     #divide the word_list based on the groups
        #     word_groups = []
        #     for i in range(10):
        #         word_groups.append([word for word, sim in self.similarity_score_dict[base_target] if groups[i] <= sim < groups[i+1]])
                
                
                
        for d in tqdm(self.full_data, desc="Generating sampled-similarity tokens (word2vec)", total=len(self.full_data)):
            base_target = d["target_true"].lstrip()  # strip leading spaces efficiently

            # Get all pairs once and reuse them
            word_sim_pairs = self.similarity_score_dict[base_target]
            words, sims = zip(*word_sim_pairs)  # This unpacks word-similarity pairs into separate lists
            
            # Convert similarities to a PyTorch tensor or Numpy array for faster processing
            similarity_scores = torch.tensor(sims)
            
            # Compute quantiles
            quantiles = torch.quantile(similarity_scores, torch.linspace(0, 1, 11))

            # Use vectorized operations to categorize words
            word_groups = []
            indices = torch.bucketize(similarity_scores, quantiles)  # This will assign each score to a bucket
            
            for i in range(10):
                # Filter words based on the bucket indices
                word_groups.append([words[j] for j, idx in enumerate(indices) if idx == i + 1])

                
            for grup in word_groups:
                #for each group create a new data point
                new_d = d.copy()
                # select 20 words from the group
                new_d["target_new"] = [ random.choice(grup) for _ in range(20)]
                new_d["similarity_group"] = grup
                new_data.append(new_d)
        
        return new_data
            


    def __generate_modify_self_similarity_data__(self):
        word2vec = api.load("word2vec-google-news-300")
        similarity_score_list = []
        for d in tqdm(
            self.full_data,
            desc="Generating self similarity tokens (word2vec)",
            total=len(self.full_data),
        ):
            base_target = d["target_true"]
            other_target = d["target_new"]
            # remove first space if present
            if other_target[0] == " ":
                other_target = other_target[1:]
            if base_target[0] == " ":
                base_target = base_target[1:]

            # compute similarity
            try:
                similarity_score = word2vec.similarity(base_target, other_target)  # type: ignore
                similarity_score_list.append(similarity_score)
            except:
                similarity_score = -100
            # save the similarity score
            d["similarity_score"] = similarity_score

        # sort full data by similarity score
        self.full_data = sorted(
            self.full_data, key=lambda x: x["similarity_score"], reverse=True
        )
        group_intervals = [
            (0, 0.1),
            (0.1, 0.2),
            (0.2, 0.3),
            (0.3, 0.4),
            (0.4, 0.5),
            (0.5, 0.6),
            (0.6, 0.7),
            (0.7, 0.725),
            (0.725, 0.750),
            (0.750, 0.775),
            (0.775, 0.8),
        ]
        # divide the data into 8 groups based on the similarity score
        for i, interval in enumerate(group_intervals):
            for d in self.full_data:
                if interval[0] <= d["similarity_score"] < interval[1]:
                    d["similarity_group"] = i
                if d["similarity_score"] >= 0.8:
                    d["similarity_group"] = -100
                if d["similarity_score"] < 0:
                    d["similarity_group"] = -100

        # Count the number of points in each group in full data
        similarity_group_count = {}
        for d in self.full_data:
            similarity_group_count[d["similarity_group"]] = (
                similarity_group_count.get(d["similarity_group"], 0) + 1
            )
        print(similarity_group_count)
        # remove points with similarity group -100
        # set the minimal size of the similarity group excluding the -100 group
        # remove the keys with -100
        similarity_group_count.pop(-100, None)
        self.minimal_size = min(similarity_group_count.values())
        return self.full_data

    def __generate_self_similarity_data__(self):
        word2vec = api.load("word2vec-google-news-300")
        similarity_score_list = []
        for d in tqdm(
            self.full_data,
            desc="Generating self similarity tokens (word2vec)",
            total=len(self.full_data),
        ):
            base_target = d["target_true"]
            other_target = d["target_new"]
            # remove first space if present
            if other_target[0] == " ":
                other_target = other_target[1:]
            if base_target[0] == " ":
                base_target = base_target[1:]

            # compute similarity
            try:
                similarity_score = word2vec.similarity(base_target, other_target)  # type: ignore
                similarity_score_list.append(similarity_score)
            except:
                similarity_score = -100
            # save the similarity score
            d["similarity_score"] = similarity_score

        # sort full data by similarity score
        self.full_data = sorted(
            self.full_data, key=lambda x: x["similarity_score"], reverse=True
        )
        # split into bins, with the highest similarity scores first
        near_high_similarity_bins = [
            self.full_data[i * 50 : 500 + (i) * 50] for i in range(10)
        ]
        high_similarity_bin = [self.full_data[500:1000]]
        remaining_bins = [
            self.full_data[i : i + 1000] for i in range(1000, len(self.full_data), 1000)
        ]
        similarity_score_bins = (
            high_similarity_bin + near_high_similarity_bins + remaining_bins
        )

        # Assign similarity_group to each data point based on the bin it falls into
        for i, bin in enumerate(similarity_score_bins):
            for d in bin:
                d["similarity_group"] = i

        # Count the number of points in each group in full data
        similarity_group_count = {}
        for d in self.full_data:
            similarity_group_count[d["similarity_group"]] = (
                similarity_group_count.get(d["similarity_group"], 0) + 1
            )
        print(similarity_group_count)
        return self.full_data

    def filter_similarity_data(self):
        if self.similarity[2] == "data-sampling":
            similarity_group = self.similarity[1]
            data_group = [d for d in self.full_data if d["similarity_group"] == similarity_group]
            # for d in data_group:
            #     #sample from the list 
            #     d["target_new"] = random.choice(d["target_new"])
            return data_group
        
        elif self.similarity[2] == "modify-self-similarity" or self.similarity[2] == "self-similarity":
            similarity_group = self.similarity[1]
            data = [d for d in self.full_data if d["similarity_group"] == similarity_group]
            if self.similarity[2] == "modify-self-similarity":
                # random sample a subset of the data of size minimal_size
                data = random.sample(data, self.minimal_size)
            return data
        else:
            raise NotImplementedError(f"Similarity type {self.similarity[2]} is not supported")

    def set_len(self, length: int):
        self.len = length

        # filter for similarity group
        if self.similarity[0]:
            data = self.filtered_data
        else:
            data = self.full_data

        # filter for length
        data = [d for d in data if d["length"] == length]
        self.prompts = [d["prompt"] for d in data]
        self.tokenized_prompts = [d["tokenized_prompt"] for d in data]
        self.targets = [d["targets"] for d in data]
        self.obj_pos = [d["obj_pos"] for d in data]
        self.first_subj_pos = [d["1_subj_pos"] for d in data]
        self.second_subj_pos = [d["2_subj_pos"] for d in data]
        self.subj_len = [d["subj_len"] for d in data]

        self.original_index = [
            i for i, d in enumerate(self.full_data) if d["length"] == length
        ]
        self.check_duplicates()

    def check_duplicates(self):
        seen = set()
        for i, d in enumerate(self.full_data):
            if d["prompt"] in seen:
                for j, d2 in enumerate(self.full_data):
                    if j == i:
                        continue
                    if d["prompt"] == d2["prompt"]:
                        if d["target_new"] == d2["target_new"]:
                            if d["target_true"] == d2["target_true"]:
                                print(f"duplicate found: {d}, {d2}")
                                return False
            seen.add(d["prompt"])
        return True


class TlensDataset(BaseDataset):
    def __init__(
        self,
        path: str,
        experiment: Literal["copyVSfact", "contextVSfact", "copyVSfact_factual"],
        model: Union[WrapHookedTransformer, str, HookedTransformer],
        slice: Optional[int] = None,
        start: Optional[int] = None,
        premise: str = "Redefine",
        similarity: Tuple[
            bool, int, Literal["word2vec", "logit", "self-similarity"]
        ] = (
            False,
            0,
            "logit",
        ),
    ):
        if isinstance(model, str):
            self.model = WrapHookedTransformer.from_pretrained(model, device="cuda")
        else:
            self.model = model
        self.model.eval()
        super().__init__(path, model, experiment, start, premise, similarity)
        # super().__init__(path, slice, start, experiment, premise, similarity)

    def _tokenize_prompt(self, prompt: str, prepend_bos: bool= False) -> torch.Tensor:
        tokens = self.model.to_tokens(prompt, prepend_bos).squeeze(0)
        assert len(tokens.shape) == 1
        return tokens

    def _tokenize_target(self, target: str, prepend_bos: bool) -> Tensor:
        if self.model.predict_with_space is False:
            # remove the first space
            target = target[1:]
        tokens = torch.tensor([self.model.to_tokens(target, prepend_bos).squeeze(0)[0]])
        # print(self.model.to_str_tokens(target,prepend_bos))
        assert (
            tokens.shape[0] == 1
        ), "tokens is not a 1D tensor with one element (the target)"
        return tokens

    def _get_similar_token(
        self,
        token_to_be_similar_str: str,
        token_to_be_similar: torch.Tensor,
        similarity_level: int,
        similarity_type: str,
        base_prompt: str,
    ) -> Tuple[str, torch.Tensor]:
        if similarity_type == "input":
            with torch.no_grad():
                token_embedding = self.model.W_E[token_to_be_similar].squeeze(0)
                embeddings = self.model.W_E

            cosine_similarity = torch.nn.functional.cosine_similarity(
                embeddings, token_embedding, dim=-1
            )
            cosine_similarity, sorted_indices = cosine_similarity.sort(descending=True)

            sorted_indices = sorted_indices[1:]
            cosine_similarity = cosine_similarity[1:]

            group = {}
            group[1] = sorted_indices[
                cosine_similarity < torch.quantile(cosine_similarity, 0.25)
            ]
            group[2] = sorted_indices[
                (cosine_similarity >= torch.quantile(cosine_similarity, 0.25))
                & (cosine_similarity < torch.quantile(cosine_similarity, 0.5))
            ]
            group[3] = sorted_indices[
                (cosine_similarity >= torch.quantile(cosine_similarity, 0.5))
                & (cosine_similarity < torch.quantile(cosine_similarity, 0.75))
            ]
            group[4] = sorted_indices[
                cosine_similarity >= torch.quantile(cosine_similarity, 0.75)
            ]

            sampled_token_idx = torch.randint(
                0, len(group[similarity_level]), (1,)
            ).item()
            sampled_token = group[similarity_level][sampled_token_idx]
            sampled_token_str = self.model.to_string(sampled_token.item())
            assert (
                sampled_token_str != token_to_be_similar_str
            ), "sampled_token_str is the same as token_to_be_similar_str"
            assert sampled_token.shape[0] == 1, "sampled_token is not a 1D tensor"
            assert len(sampled_token.shape) == 2, "sampled_token is not a 2D tensor"
            assert isinstance(
                sampled_token_str, str
            ), "sampled_token_str is not a string"
            return sampled_token_str, sampled_token.unsqueeze(0).unsqueeze(0)
        elif similarity_type == "output":
            raise NotImplementedError
        else:
            raise ValueError("similarity_type must be either 'input' or 'output'")

    @abstractmethod
    def _get_vocab_size(self) -> int:
        return self.model.vocab_size

    @abstractmethod
    def _to_string_token(self, token_id: int) -> str:
        return self.model.to_string(torch.tensor(token_id))

    def get_best_string_token_predictions(self, prompt: str, n: int) -> List[str]:
        raise NotImplementedError

    def get_token_per_similarity_level(
        self, prompt: str, best_string_token_predictions: List[str]
    ) -> Dict[int, List[str]]:
        raise NotImplementedError

    def compute_similarity_word2vec(
        self, base_target: str, word2vec
    ) -> List[Tuple[str, float]]:
        raise NotImplementedError


class HFDataset(BaseDataset):
    def __init__(
        self,
        model: Union[AutoModelForCausalLM, str],
        tokenizer,
        path: str,
        experiment: Literal["copyVSfact", "contextVSfact"],
        slice: Optional[int] = None,
        premise: str = "Redefine:",
        similarity: Tuple[
            bool, int, Literal["word2vec", "logit", "self-similarity"]
        ] = (
            False,
            0,
            "logit",
        ),
        family_name: str = "gpt2",
    ):
        if isinstance(model, str):
            self.model = AutoModelForCausalLM.from_pretrained(
                model,
                torch_dtype=torch.bfloat16,
            )
            self.model = self.model.cpu()

        else:
            self.model = model
        self.tokenizer = tokenizer
        super().__init__(
            path=path,
            slice=slice,
            experiment=experiment,
            premise=premise,
            similarity=similarity,
            family_name=family_name,
        )

    def reset(
        self,
        similarity: Tuple[
            bool, int, Literal["word2vec", "logit", "self-similarity"]
        ] = None,
    ):
        super().reset(similarity=similarity)

    def _tokenize_prompt(self, prompt: str, prepend_bos: bool) -> torch.Tensor:
        # bos_token = self.tokenizer.bos_token
        # if prepend_bos:
        #     prompt = bos_token + prompt
        tokens = self.tokenizer.encode(
            prompt, return_tensors="pt", add_special_tokens=True
        ).squeeze(0)
        assert len(tokens.shape) == 1
        return tokens

    def _tokenize_target(self, target: str, prepend_bos: bool) -> torch.Tensor:
        tokens = self.tokenizer.encode(
            target, return_tensors="pt", add_special_tokens=False
        ).squeeze(0)
        assert len(tokens.shape) == 1
        if tokens.shape[0] > 1:
            tokens = tokens[0]
            tokens = tokens.unsqueeze(0)
        return tokens

    def compute_similarity_word2vec(
        self, base_target: str, word2vec, other_target: str
    ) -> List[Tuple[str, float]]:
        similarity = []
        # remove the first space
        base_target = base_target[1:]

        if self.similarity_score_dict.get(base_target) is not None:
            return self.similarity_score_dict[base_target]

        # print(self.tokenizer.encode(" C"))
        for token in range(self.tokenizer.vocab_size):
            str_token = self.tokenizer.decode(token)
            if str_token[0] == " ":
                space_token = str_token
                str_token = str_token[1:]
            else:
                space_token = " " + str_token
            # if str_token == other_target:
            #     print("found")
            try:
                similarity.append(
                    (space_token, word2vec.similarity(base_target, str_token))
                )
            except KeyError:
                if " " + str_token == other_target:
                    print("other_target", other_target, " is not in the w2v vocab")

        # save the similarity score
        self.similarity_score_dict[base_target] = similarity
        # similarity_len = len(similarity)
        # print(similarity_len)
        return similarity


#     def compute_similarity_word2vec(self, base_target, word2vec, other_target):
#         base_target = base_target[1:]

#         with Pool(processes=4) as pool:  # Adjust the number of processes as needed
#             results = pool.starmap(
#                 compute_similarity_for_token,
#                 [(base_target, word2vec, token, self.tokenizer) for token in range(self.tokenizer.vocab_size)]
#             )

#         # Filter out None results and return
#         return [result for result in results if result is not None]


# def compute_similarity_for_token(base_target, word2vec, token, tokenizer):
#     str_token = tokenizer.decode(token)
#     if str_token[0] == " ":
#         space_token = str_token
#         str_token = str_token[1:]
#     else:
#         space_token = " " + str_token

#     try:
#         return space_token, word2vec.similarity(base_target, str_token)
#     except KeyError:
#         return None


class SampleDataset:
    def __init__(self, path: str, model, save_path: str, tokenizer: Optional[object]):
        self.data = json.load(open(path))
        self.model = model
        self.save_path = save_path
        self.checkpoint_size = 500
        if type(model) == WrapHookedTransformer:
            self.model_type = "WrapHookedTransformer"
            self.tokenizer = model.tokenizer
        else:
            self.model_type = "AutoModelForCausalLM"
            try:
                self.tokenizer = tokenizer
            except AttributeError:
                raise ValueError("With HuggingFace models, you must pass a tokenizer")

    def sample(self, size: int = 10000):
        if type(self.model) == WrapHookedTransformer:
            self.sample_dataset_tlens(size)
        else:
            self.sample_dataset_hf(size)

    def sample_dataset_tlens(self, size: int):
        # random.seed(42)
        new_data, index = self.load_from_checkpoint()

        random.shuffle(self.data)
        self.data = self.data[index:]
        size = size - len(new_data)
        new_data = []
        random.shuffle(self.data)
        with tqdm(total=size) as pbar:
            for i, d in enumerate(self.data):
                if i % self.checkpoint_size == 0:
                    self.checkpoint(i, new_data)
                # empty_prompt = d["template"].format("Redefine", self.model.tokenizer.pad_token)
                empty_prompt = d["base_prompt"]
                if self.model.predict(empty_prompt)[1][0] == d["target_true"]:
                    new_data.append(d)
                    if len(new_data) == size:
                        break
                pbar.update(len(new_data) - pbar.n)
            self.data = new_data

    def sample_dataset_hf(self, size: int):
        # random.seed(42)
        new_data, index = self.load_from_checkpoint()

        random.shuffle(self.data)
        self.data = self.data[index:]
        print(len(self.data), index)

        with tqdm(total=size) as pbar:
            pbar.update(len(new_data))
            for i, d in enumerate(self.data):
                if i % self.checkpoint_size == 0:
                    self.checkpoint(i, new_data)
                empty_prompt = d["base_prompt"]
                # encode the prompt
                input_ids = self.tokenizer.encode(empty_prompt, return_tensors="pt")  # type: ignore
                input_ids = input_ids.to(self.model.device)  # type: ignore
                target_true = self.tokenizer.encode(
                    d["target_true"], return_tensors="pt", add_special_tokens=False
                )  # type: ignore
                # predict the next token
                logits = self.model(input_ids)["logits"][0, -1, :].cpu()
                # get the index of the predicted token
                index = logits.argmax()
                # check if the predicted token is the target

                if index in target_true:
                    new_data.append(d)
                    if len(new_data) == size:
                        break
                pbar.update(len(new_data) - pbar.n)
            self.data = new_data

    def save(self):
        json.dump(self.data, open(self.save_path, "w"), indent=2)

    def checkpoint(self, size: int, data: list):
        # create checkpoint folder

        if not os.path.isdir("../data/checkpoints"):
            os.mkdir("../data/checkpoints")
        # save the data
        # split save path
        save_path = self.save_path.split("/")[2]
        json.dump(data, open(f"../data/checkpoints/{save_path}", "w"), indent=2)

    def load_from_checkpoint(self):
        save_path = self.save_path.split("/")[2]
        # check if the checkpoint exists
        if not os.path.isfile(f"../data/checkpoints/{save_path}"):
            return [], 0
        print("Starting from checkpoint")
        data = json.load(open(f"../data/checkpoints/{save_path}"))
        # get index of the last data point
        index = len(data) - 1
        if index < 0:
            return [], 0
        return data, index


class DatasetGenerator:
    def __init__(self, path):
        self.data = json.load(open(path))

    def generate_dataset(self, model, lenghts=[17, 19, 23]):
        my_data = []
        for i, d in tqdm(
            enumerate(self.data), total=len(self.data), desc="Generating dataset"
        ):
            target_new = " " + d["requested_rewrite"]["target_true"]["str"]
            target_true = " " + d["requested_rewrite"]["target_new"]["str"]
            if i % 50 == 0:
                unique_strs = set(json.dumps(d) for d in my_data)
                my_data = [json.loads(s) for s in unique_strs]
                print(len(my_data))
                # if len(my_data) > 1000:
                #     break
            for p in d["attribute_prompts"]:
                template = "Redefine: " + p + "{}" + ". " + p
                # find position of {} in template
                if (
                    len(model.to_str_tokens(template.format(model.tokenizer.pad_token)))
                    not in lenghts
                ):
                    continue
                try:
                    obj_pos = (
                        model.to_str_tokens(
                            template.format(model.tokenizer.pad_token)
                        ).index(".")
                        - 1
                    )
                except:  # noqa: E722
                    continue
                if target_true in template:
                    continue
                prediction = model.predict(template.format(model.tokenizer.pad_token))[
                    1
                ][0]
                copy_prediction = model.predict(template.format(target_new))[1][0]
                if prediction == target_true and copy_prediction == target_new:
                    my_data.append(
                        {
                            "prompt": p,
                            "template": template,
                            "prediction": prediction,
                            "copy_prediction": copy_prediction,
                            "target_true": target_true,
                            "target_new": target_new,
                            "length": len(
                                model.to_str_tokens(
                                    template.format(model.tokenizer.pad_token)
                                )
                            ),
                            "lenght_copy": len(
                                model.to_str_tokens(template.format(target_new))
                            ),
                            "obj_pos": obj_pos,
                        }
                    )
            for p in d["neighborhood_prompts"]:
                template = "Redefine: " + p + "{}" + ". " + p
                # find position of {} in template
                if (
                    len(model.to_str_tokens(template.format(model.tokenizer.pad_token)))
                    not in lenghts
                ):
                    continue
                try:
                    obj_pos = (
                        model.to_str_tokens(
                            template.format(model.tokenizer.pad_token)
                        ).index(".")
                        - 1
                    )
                except:  # noqa: E722
                    continue
                if target_true in template:
                    continue
                prediction = model.predict(template.format(model.tokenizer.pad_token))[
                    1
                ][0]
                copy_prediction = model.predict(template.format(target_new))[1][0]
                if prediction == target_true and copy_prediction == target_new:
                    # check if is a duplicate

                    my_data.append(
                        {
                            "prompt": p,
                            "template": template,
                            "prediction": prediction,
                            "copy_prediction": copy_prediction,
                            "target_true": target_new,
                            "target_new": target_true,
                            "length": len(
                                model.to_str_tokens(
                                    template.format(model.tokenizer.pad_token)
                                )
                            ),
                            "lenght_copy": len(
                                model.to_str_tokens(template.format(target_new))
                            ),
                            "obj_pos": obj_pos,
                        }
                    )

        print(
            "Number of examples:", len(my_data), "Number of possible lengths:", lenghts
        )
        self.my_data = my_data

    def save(self, path):
        json.dump(self.my_data, open(path, "w"), indent=2)
