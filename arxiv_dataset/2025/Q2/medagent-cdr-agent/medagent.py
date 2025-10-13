import json
import numpy as np
import random
from scipy.stats import norm
from pyarrow.lib import asarray
from scipy.spatial import distance
from sklearn.mixture import GaussianMixture
from collections import Counter
import cdr_code_no_io
import cdr_io_criteria
import time

class MedAgent:

    def __init__(self,
                 core_model,
                 cdr_info_path,
                 cdr_config,
                 anomaly_truncate_keep_ratio=0.9,
                 anomaly_detection_rep=10,
                 anomaly_detection_freq_factor=2.0,
                 anomaly_sampling=False,
                 irr_info_robustness=False,
                 cut_number=50,
                 cut_ratio=0.5,
                 silence=True,
                 num_candidate_cdrs = 1,
                 anomaly_cdr=False,
                 ):
        self.core_model = core_model
        self.silence = silence
        self.num_candidate_cdrs = num_candidate_cdrs
        self.anomaly_cdr = anomaly_cdr
        self.cdr_config = cdr_config
        self.irr_info_robustness=irr_info_robustness
        self.cut_number = cut_number
        self.cut_ratio = cut_ratio
        self.anomaly_detection_rep = anomaly_detection_rep
        self.anomaly_truncate_keep_ratio = anomaly_truncate_keep_ratio
        self.anomaly_detection_freq_factor = anomaly_detection_freq_factor
        self.anomaly_sampling = anomaly_sampling
        # Prepare cdr info
        with open(cdr_info_path, 'r') as file:
            self.cdrs_raw = json.load(file)
        self.cdrs_text = []
        self.cdrs_embedding = []
        for cdr in self.cdrs_raw:
            if self.cdr_config == 'var_meaning':
                cdr_text = "\n".join(f"{key}: {value['meaning']}" for key, value in cdr['variables'].items())
            elif self.cdr_config == 'var_only':
                cdr_text = "\n".join(f"{var}" for var in cdr['variables'].keys())
            elif self.cdr_config == 'rule_text':
                cdr_text = cdr['rule_text']
            elif self.cdr_config == 'rule_text_augmented':
                cdr_text = cdr['rule_text_augmented']
            self.cdrs_text.append({
                "label": cdr["label"],
                "cdr_text": cdr_text,
                "cdr_func_name": cdr["func_name"],
                "has_io_criteria": cdr["has_io_criteria"],
                "cdr_io_func_name": cdr["io_criteria_func_name"],
            })
            self.cdrs_embedding.append({
                "label": cdr["label"],
                "cdr_embedding": self.core_model.embed(cdr_text)
            })

    def fit_gmm_bic(self, data):
        data = data.reshape(-1, 1)
        bic_scores = []
        models = []
        n_components_range = range(1, 10)  # Try 1 to 9 clusters
        for n_components in n_components_range:
            gmm = GaussianMixture(n_components=n_components, random_state=42)
            gmm.fit(data)
            bic_scores.append(gmm.bic(data))
            models.append(gmm)
        best_model = models[np.argmin(bic_scores)]
        best_k = np.argmin(bic_scores)
        labels = best_model.predict(data)
        highest_mean_cluster = np.argmax(best_model.means_)
        highest_mean_indices = np.where(labels == highest_mean_cluster)[0]
        highest_mean_values = data[highest_mean_indices].flatten()

        return highest_mean_indices, highest_mean_values, best_k

    '''def anomaly_detection(self, data, thres=0.025):
        mean = np.mean(data)
        std = np.std(data)
        upper_bound = norm.ppf(1-thres, loc=mean, scale=std)
        anomaly_indices = np.where(data > upper_bound)[0]
        anomaly_values = data[anomaly_indices]
        if len(anomaly_indices) > 0:
            anomaly_scores = np.abs((anomaly_values - mean) / std)
            sorted_indices = np.argsort(-anomaly_values)
            anomaly_indices = anomaly_indices[sorted_indices]
            anomaly_values = anomaly_values[sorted_indices]
            anomaly_scores = anomaly_scores[sorted_indices]
        else:
            # If no anomalies, return the index and value of the largest element
            #max_index = np.argmax(data)
            #max_value = data[max_index]
            #max_score = np.abs((max_value - mean) / std)
            #return np.array([max_index]), np.array([max_value]), np.array([max_score])
            return np.array([]), np.array([]), np.array([])
        return anomaly_indices, anomaly_values, anomaly_scores'''

    def anomaly_detection(self, data, thres=0.025, rep=1):
        mean = np.mean(data)
        std = np.std(data)
        upper_bound = norm.ppf(1-thres, loc=mean, scale=std)
        anomaly_indices = np.where(data > upper_bound)[0]
        anomaly_values = data[anomaly_indices]
        if len(anomaly_indices) > 0:
            anomaly_scores = np.abs((anomaly_values - mean) / std)
            sorted_indices = np.argsort(-anomaly_values)
            anomaly_indices = anomaly_indices[sorted_indices]
            anomaly_values = anomaly_values[sorted_indices]
            anomaly_scores = anomaly_scores[sorted_indices]
        else:
            # # If no anomalies, return the index and value of the largest element
            # max_index = np.argmax(data)
            # max_value = data[max_index]
            # max_score = np.abs((max_value - mean) / std)
            # return np.array([max_index]), np.array([max_value]), np.array([max_score])
            return np.array([]), np.array([]), np.array([])
        if self.anomaly_sampling:
          anomaly_indices = anomaly_indices // rep
          count = Counter(anomaly_indices)
          anomaly_indices = [key for key, value in count.items() if value > int(rep / self.anomaly_detection_freq_factor)]
          anomaly_indices = np.asarray(anomaly_indices)
        return anomaly_indices, anomaly_values, anomaly_scores

    def find_cdr(self, clinical_notes):
        if self.irr_info_robustness:
            # Partition of the clinical note
            notes_sentences = clinical_notes.split(' ')
            cut_length = int(self.cut_ratio * len(notes_sentences))
            note_pieces = [notes_sentences[random.randint(0, len(notes_sentences) - cut_length):][:cut_length] for _ in range(self.cut_number)]
            embedding_pieces = []
            for i in range(len(note_pieces)):
                embedding_pieces.append(self.core_model.embed(" ".join(note_pieces[i])))
            sim_scores = []
            for record in self.cdrs_embedding:
                sim_scores_pieces = []
                for i in range(len(embedding_pieces)):
                    sim_scores_pieces.append(1-distance.cosine(embedding_pieces[i], record["cdr_embedding"]))
                sim_scores_pieces = np.asarray(sim_scores_pieces)
                indices, values, best_k = self.fit_gmm_bic(sim_scores_pieces)
                sim_scores.append(np.mean(values))
        else:
            clinical_notes_embedding = self.core_model.embed(clinical_notes)
            sim_scores = []
            for record in self.cdrs_embedding:
                sim_scores.append(1 - distance.cosine(clinical_notes_embedding, record["cdr_embedding"]))
        '''if self.anomaly_cdr:
            sim_scores = np.asarray(sim_scores)
            candidate_cdrs_idx, candidate_cdrs_sim_score, candidate_cdrs_anomaly_score = self.anomaly_detection(sim_scores)'''
        if self.anomaly_cdr:
          if self.anomaly_sampling:
            ### Added features
            notes_sentences = clinical_notes.split(' ')
            cut_length = int(self.anomaly_truncate_keep_ratio * len(notes_sentences))
            #note_pieces = [notes_sentences[random.randint(0, len(notes_sentences) - cut_length):][:cut_length] for _ in range(self.anomaly_detection_rep)]
            random_indices = [random.randint(0, len(notes_sentences) - cut_length) for _ in range(self.anomaly_detection_rep)]
            note_pieces = [notes_sentences[r:r + cut_length] for r in random_indices]
            #print([" ".join(note_pieces[i]) for i in range(len(note_pieces))])
            embedding_pieces = []
            for i in range(len(note_pieces)):
                embedding_pieces.append(self.core_model.embed("\n".join(note_pieces[i])))
            
            sim_scores_labels = []
            # sim_scores = []
            #for record in self.cdrs_embedding:
            #    for i in range(len(embedding_pieces)):
            #        sim_scores.append(1-distance.cosine(embedding_pieces[i], record["cdr_embedding"]))
            # sim_scores = np.asarray(sim_scores)
            from scipy.spatial.distance import cdist
            cdr_embeddings = np.array([record["cdr_embedding"] for record in self.cdrs_embedding])
            embedding_pieces = np.array(embedding_pieces)
            sim_scores = 1 - cdist(embedding_pieces, cdr_embeddings, metric='cosine')
            sim_scores = sim_scores.flatten()
            ###
            
            # candidate_cdrs_idx, candidate_cdrs_sim_score, candidate_cdrs_anomaly_score = self.anomaly_detection(sim_scores)
            candidate_cdrs_idx, candidate_cdrs_sim_score, candidate_cdrs_anomaly_score = self.anomaly_detection(sim_scores, rep=self.anomaly_detection_rep)
          else:
            sim_scores = np.asarray(sim_scores)
            candidate_cdrs_idx, candidate_cdrs_sim_score, candidate_cdrs_anomaly_score = self.anomaly_detection(sim_scores)
        else:
            candidate_cdrs_sim_score = np.flip(np.sort(sim_scores)[-self.num_candidate_cdrs:])
            candidate_cdrs_idx = np.flip(np.argsort(sim_scores)[-self.num_candidate_cdrs:])
            candidate_cdrs_anomaly_score = None
        candidate_cdrs = []
        if len(candidate_cdrs_idx) > 0:
            for idx in candidate_cdrs_idx:
                candidate_cdrs.append(self.cdrs_embedding[idx]['label'])

        return candidate_cdrs, candidate_cdrs_sim_score, candidate_cdrs_anomaly_score

    def json_to_text(self, data):
        lines = []
        for key, value in data.items():
            if isinstance(value, dict):
                nested_text = self.json_to_text(value)
                lines.append(f"{key}: \n{nested_text}")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)

    def convert_value(self, value, type_name):
        type_dict = {
            'int': int,
            'float': float,
            'string': str,
            'bool': lambda x: x.lower() in ['true', '1', 't', 'y', 'yes']  # Converts 'true', '1', etc., to True
        }
        try:
            if type_name in type_dict:
                return type_dict[type_name](value)
            else:
                raise ValueError(f"Unsupported type specified: {type_name}")
        except KeyError:
            print(f"No conversion function available for type: {type_name}")
        except ValueError as e:
            print(f"Conversion to {type_name} failed for value '{value}': {str(e)}")
        except Exception as e:
            print(f"Error: Other CDR execution error - {e}")

    def process(self, clinical_notes):
        if not self.silence:
            print('---------------MedAgent in Progress---------------')
            print('Clinical notes: {}'.format(clinical_notes))
        start_time = time.time()
        clinical_decisions = []
        # Find relevant cdrs based on cosine similarity
        relevant_cdrs, sim_scores, anomaly_scores = self.find_cdr(clinical_notes)
        intermediate_time = time.time()
        if not self.silence:
            if self.anomaly_cdr:
                print('Find relevant CDRs: {}'.format(relevant_cdrs))
                print('Associated anomaly scores: {}'.format(anomaly_scores))
            else:
                print('Find relevant CDRs: {}'.format(relevant_cdrs))

        exclude_cdrs_idx = []
        for i in range(len(relevant_cdrs)):
            cdr = relevant_cdrs[i]
            if not self.silence:
                print('** Processing {} CDR **'.format(cdr))
            # Find the cdr function name
            cdr_func_name = None
            for item in self.cdrs_text:
                if item['label'] == cdr:
                    cdr_func_name = item['cdr_func_name']
                    cdr_io_func_name = item['cdr_io_func_name']
                    cdr_has_io_criteria = item['has_io_criteria']
                    break
            # Find the variable values in the clinical notes
            cdr_raw = None
            for item in self.cdrs_raw:
                if item['label'] == cdr:
                    cdr_raw = item
                    break
            variables_json = cdr_raw['variables']
            variables_info = self.json_to_text(variables_json)      # cdr variable definition and type in text
            from prompt_library import variable_retrieval
            prompt = variable_retrieval.format(variable_descriptions=variables_info, clinical_note=clinical_notes)
            response_raw = self.core_model.generate(prompt)[0]
            variables = {}
            try:
                response = response_raw.split('[')[1]
                response = response.split(']')[0]
                response = response.strip().split(', ')
                for line in response:
                    if ': ' in line:
                        key, value = line.split(': ', 1)
                        variable_type = cdr_raw['variables'][key]['type']
                        if not value in ['None', 'Unknown', '___', 'unknown', 'none']:
                            value = self.convert_value(value, variable_type)
                        else:
                            value = None # will be replaced with default value in the cdr code execution
                        variables[key] = value
            except ValueError as e:
                print(f"Error: Invalid value - {e}")
            except TypeError as e:
                print(f"Error: Type mismatch - {e}")
            except KeyError as e:
                print(f"Error: Invalid key - {e}")
            except IndexError:
                print("Error: Incorrect response format!")
            if not self.silence:
                print('Find variables:')
                for key, value in variables.items():
                    print(f" - {key}: {value}")
            
            # Check io criteria
            if cdr_has_io_criteria:
                cdr_io_func = getattr(cdr_io_criteria, cdr_io_func_name, None)
                if callable(cdr_io_func):
                    try:
                        result = cdr_io_func(variables)
                    except ValueError as e:
                        print(f"Error: Invalid value - {e}")
                    except TypeError as e:
                        print(f"Error: Type mismatch - {e}")
                    except Exception as e:
                        print(f"Error: Other CDR execution error - {e}")
                else:
                    print(f"No io function named {cdr_io_func_name} found in the toolbox")
                if result == "Exclude":
                    print(f" {cdr_io_func_name} excluded after io checking")
                    exclude_cdrs_idx.append(i)
                    continue
                
            # Call the cdr function
            result = None
            cdr_func = getattr(cdr_code_no_io, cdr_func_name, None)
            if callable(cdr_func):
                try:
                    result = cdr_func(variables)
                except ValueError as e:
                    print(f"Error: Invalid value - {e}")
                except TypeError as e:
                    print(f"Error: Type mismatch - {e}")
                except Exception as e:
                    print(f"Error: Other CDR execution error - {e}")
            else:
                print(f"No function named {cdr_func_name} found in the toolbox")
            if not self.silence:
                print('Clinical decision: {}'.format(result[0]))
            clinical_decisions.append({
                'cdr': cdr,
                'sim_score': sim_scores[i].item(),
                'model response': response_raw,
                'variables': variables,
                'decision': result[0]
            })
            
        # filter excluded cdrs
        final_relevant_cdrs = []
        for i in range(len(relevant_cdrs)):
            if i not in exclude_cdrs_idx:
                final_relevant_cdrs.append(relevant_cdrs[i])
                
        end_time = time.time()
        if len(final_relevant_cdrs) == 0:
            clinical_decisions.append({
                'cdr': 'no applicable cdr',
                'sim_score': None,
                'model response': None,
                'variables': None,
                'decision': None
            })

        return clinical_decisions, intermediate_time - start_time, end_time - intermediate_time

