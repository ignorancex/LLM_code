import numpy as np
import random
from itertools import combinations
from soar.llm_utils import prompt2token_chat, save_pkl_secure_force,save_pkl_secure, get_number_of_solved_tasks, format_all_generation
from soar.sandbox.execute_code_less_safe import check_solutions
from soar.prompt import (
    get_repair_prompt,
    prompt_repair_cot_fewshot_crossover_v1, 
    prompt_repair_sol_v1
)
import copy
from tqdm import trange
from soar.api import get_multiple_completions,get_multiple_completions_multiple_client


class REX:
    def __init__(self, archive:dict, 
                 data2test:dict,
                 path_save:str,
                 sampling_method:str="uniform",
                 crossover:bool=False,
                 correctness=None,
                 C: float = 20,
                 N_budget:int = 512,
                 temperature:float = 1.,
                 n_completions:int = 1,
                 openai = False,
                 client=None,cfg_generation=None,
                 max_workers=5, gpt_mini_mode=False,
                 max_tokens= 4096,
                 sglang = True,
                 split="train"):
        """
        format archive = {task_id:[{...},] }
        sampling_method = "uniform" "uniform_correctness" "REX" "REX_ablation_discount" "coverage_test_units"


        correctness = "zero", "low", "medium", "max",  (use with sampling_method = "uniform_correctness") #"uncompilable"
        """
        self.archive = copy.deepcopy(archive)
        self.list_task_id = list(self.archive.keys())
        self.data2test = data2test
        self.path_save = path_save
        self.sampling_method = sampling_method
        self.C = C
        self.crossover = crossover
        self.correctness = correctness
        self.list_correctness = ["zero", "low", "medium", "max"]
        self.init_archive()
        self.N_budget = N_budget
        self.temperature = temperature
        self.n_completions = n_completions
        self.openai = openai
        self.client = client
        self.cfg_generation = cfg_generation
        self.max_workers = max_workers
        self.gpt_mini_mode = gpt_mini_mode
        self.split = split
        self.max_tokens = max_tokens
        self.sglang = sglang

    def init_archive(self):
        """set N=0 for all programs in archive and add unique_id"""
        self.unique_id = 0
        max_unique_id = 0
        self.restart_checkpoint = False
        # Determine if we are restarting from a checkpoint by checking if any program already has a 'unique_id'
        self.restart_checkpoint = any(
            'unique_id' in prog for task_id in self.list_task_id for prog in self.archive[task_id]
        )
        for task_id in self.list_task_id:
            for i in range(len(self.archive[task_id])):
                if self.restart_checkpoint:
                    id = self.archive[task_id][i]['unique_id']
                    if int(id) >= max_unique_id:
                        self.unique_id = int(id) + 1
                else:
                    self.archive[task_id][i]['unique_id'] = str(self.unique_id)
                    self.unique_id += 1
                    self.archive[task_id][i]['N'] = 0
                    self.archive[task_id][i]["task_id"] = task_id
                    self.archive[task_id][i]["parents"] = []
                    self.archive[task_id][i]["type"] = "initial_solution"
                    if self.restart_checkpoint:
                        id = self.archive[task_id][i]['unique_id']
                        if int(id) >= max_unique_id:
                            self.unique_id = int(id)+1
                    else:
                        self.archive[task_id][i]['unique_id'] = str(self.unique_id)
                        self.unique_id += 1
                        self.archive[task_id][i]['N'] = 0
                        self.archive[task_id][i]["task_id"] = task_id
                        self.archive[task_id][i]["parents"] = []
                        self.archive[task_id][i]["type"] = "initial_solution"

        if self.restart_checkpoint:
            self.initial_problem = len([self.archive[task_id][i]["type"] for i in range(len(self.archive[task_id])) if self.archive[task_id][i]["type"] == "initial_solution"])
        else:
            self.initial_problem = len(self.archive[task_id])
    def save_archive(self,force=True):
        if force: 
            save_pkl_secure_force(self.path_save, {"dict_response":self.archive})
        else:
            save_pkl_secure(self.path_save, {"dict_response":self.archive})

    def heuristic_train(self, response_dict):
        return np.mean(response_dict["correct_train_input"])

    def thompson_sampling(self, response_dict) -> float:
        h = self.heuristic_train(response_dict)
        return np.random.beta(1 + self.C * h, 1 + self.C * (1 - h) + response_dict['N'])
    
    def thompson_sampling_ablation_discount(self, response_dict) -> float:
        h = self.heuristic_train(response_dict)
        return np.random.beta(1 + self.C * h, 1 + self.C * (1 - h))


    def get_category_correctness(self, response_dict):
        cat = np.mean(response_dict["correct_train_input"])
        if cat==0:
            # if all([isinstance(i,str) for i in response_dict["predicted_train_output"]]):
            #     return "uncompilable"
            # else:
            return "zero"   
        elif cat<=0.34:
            return "low"
        elif cat <0.98:
            return "medium"
        else:
            return "max"
        
    def uniforme_difficulty(self, response_dict):
        """return a task_id with uniform probability"""
        if self.correctness == self.get_category_correctness(response_dict):
            return 1
        else:
            return 0 
        
    def calculate_combined_coverage(self,prog1, prog2):
        return np.mean([a or b for a, b in zip(prog1['correct_train_input'], prog2['correct_train_input'])])

    def get_all_descendants(self, id, archive_tasks):
        descendants = []
        for prog in archive_tasks:
            if id in prog['parents']:
                descendants.append(prog['unique_id'])
                descendants.extend(self.get_all_descendants(prog['unique_id'], archive_tasks))
        return list(set(descendants))  # Remove duplicates

    def get_all_ucb_quality(self, id, archive_tasks):
        list_descendants = self.get_all_descendants(id, archive_tasks)
        list_parent_descendants = [id] + list_descendants
        list_quality = []
        for prog in archive_tasks:
            if prog['unique_id'] in list_parent_descendants:
                list_quality.append(self.heuristic_train(prog))
        return np.mean(list_quality)

    def coverage_test_units_sampling(self, task_id):

        # max_coverage = -1
        # best_pairs = []
        list_all_pair = []
        list_fitness = []
        archive_task_id = self.archive[task_id]
        list_idx= [i for i in range(len(archive_task_id))]
        if len(list_idx) < 2:
            return [self.archive[task_id][0], self.archive[task_id][0]]

        # Generate all possible unique pairs
        for idx_prog1, idx_prog2 in combinations(list_idx, 2):
            coverage = self.calculate_combined_coverage(archive_task_id[idx_prog1], archive_task_id[idx_prog2])
            list_all_pair.append((idx_prog1, idx_prog2))
            freq = archive_task_id[idx_prog1]["N"] + archive_task_id[idx_prog2]["N"]
            weight = 1 / (freq + 1)
            fitness = coverage + weight
            list_fitness.append(fitness)

        total_weight = sum(fitness for fitness in list_fitness)
        probabilities = [weight / total_weight for weight in list_fitness]
        selected_pair = random.choices(
            list_all_pair,
            weights=probabilities,
            k=1
        )[0]

        # Update count tracker
        (idx1,idx2) = selected_pair
        self.archive[task_id][idx1]["N"] += 1/2
        self.archive[task_id][idx2]["N"] += 1/2
        selected_pair_program = [self.archive[task_id][idx1], self.archive[task_id][idx2]]
        return selected_pair_program
    
    def coverage_test_units_thompson_sampling(self, task_id):

        # max_coverage = -1
        # best_pairs = []
        list_all_pair = []
        list_theta = []
        archive_task_id = self.archive[task_id]
        list_idx= [i for i in range(len(archive_task_id))]
        if len(list_idx) < 2:
            return [self.archive[task_id][0], self.archive[task_id][0]]

        # Generate all possible unique pairs
        for idx_prog1, idx_prog2 in combinations(list_idx, 2):
            coverage = self.calculate_combined_coverage(archive_task_id[idx_prog1], archive_task_id[idx_prog2])
            
            list_all_pair.append((idx_prog1, idx_prog2))
            freq = (archive_task_id[idx_prog1]["N"] + archive_task_id[idx_prog2]["N"]) / 2

            theta= np.random.beta(1 + self.C * coverage, 1 + self.C * (1 - coverage)+ freq)
            list_theta.append(theta)
        idx_selected_program = np.argmax(list_theta)
        selected_pair = list_all_pair[idx_selected_program]

        # Update count tracker
        (idx1,idx2) = selected_pair
        self.archive[task_id][idx1]["N"] += 1
        self.archive[task_id][idx2]["N"] += 1
        selected_pair_program = [self.archive[task_id][idx1], self.archive[task_id][idx2]]
        return selected_pair_program
    

    def coverage_test_units_ucb_sampling(self, task_id):
        """test, need some optimization"""
        list_all_pair = []
        list_ucb_scores = []
        archive_task_id = self.archive[task_id]
        list_idx = [i for i in range(len(archive_task_id))]
        total_samples = len(archive_task_id) - self.initial_problem
        if len(list_idx) < 2:
            return [self.archive[task_id][0], self.archive[task_id][0]]
        # Generate all possible unique pairs
        for idx_prog1, idx_prog2 in combinations(list_idx, 2):
            list_all_pair.append((idx_prog1, idx_prog2))
            
            # Calculate average number of times this pair has been sampled
            pair_samples = (archive_task_id[idx_prog1]["N"] + archive_task_id[idx_prog2]["N"]) / 2
            
            
            coverage = self.calculate_combined_coverage(archive_task_id[idx_prog1], archive_task_id[idx_prog2]) # /!\ + add average child coverage
            coverage += self.get_all_ucb_quality(archive_task_id[idx_prog1]["unique_id"], archive_task_id) + self.get_all_ucb_quality(archive_task_id[idx_prog2]["unique_id"], archive_task_id)
            coverage /= 3

            # UCB score calculation

            exploitation = coverage 

            exploration = np.sqrt(2*np.log(total_samples + 1) / (pair_samples + 1))
            ucb_score = exploitation + exploration #self.C * exploration

            list_ucb_scores.append(ucb_score)

        idx_selected_program = np.argmax(list_ucb_scores)
        selected_pair = list_all_pair[idx_selected_program]

        # Update count tracker
        idx1, idx2 = selected_pair
        self.archive[task_id][idx1]["N"] += 1
        self.archive[task_id][idx2]["N"] += 1
        selected_pair_program = [self.archive[task_id][idx1], self.archive[task_id][idx2]]
        
        return selected_pair_program

    def sample_program_2_refine(self,task_id) -> list[str]:
        """sample a list of program from archive to refine given task_id,
        should return one or two program [self.archive[task_id][i]] or [self.archive[task_id][i],self.archive[task_id][j]] """
        if self.crossover:
            n_program2sample=2
        else:
            n_program2sample=1
        if self.sampling_method == "uniform":
            return np.random.choice(self.archive[task_id], size = n_program2sample)
        elif self.sampling_method == "uniform_correctness":
            list_proba = np.array([self.uniforme_difficulty(i) for i in self.archive[task_id]])
            sum_proba = sum(list_proba)
            if sum_proba == 0:
                return np.random.choice(self.archive[task_id], size = n_program2sample)
            else:
                return np.random.choice(self.archive[task_id], p=list_proba/sum_proba, size = n_program2sample)
            
        elif self.sampling_method == "REX":
            theta = [self.thompson_sampling(p) for p in self.archive[task_id]]
            # idx_selected_program = np.argmax(theta)
            idx_selected_programs = np.argsort(theta)[-n_program2sample:][::-1]

            return [self.archive[task_id][i] for i in idx_selected_programs]
        
        elif self.sampling_method == "REX_ablation_discount":
            """REX without discounting the number of times a program has been selected"""
            theta = [self.thompson_sampling_ablation_discount(p) for p in self.archive[task_id]]
            idx_selected_programs = np.argsort(theta)[-n_program2sample:][::-1]

            return [self.archive[task_id][i] for i in idx_selected_programs]

        elif self.sampling_method == "coverage_test_units":
            """take sample that cover as much as possible the test units space"""
            return self.coverage_test_units_sampling(task_id)
        
        elif self.sampling_method == "coverage_test_units_thompson":
            """take sample that cover as much as possible the test units space"""
            return self.coverage_test_units_thompson_sampling(task_id)
        
        elif self.sampling_method == "coverage_test_units_ucb":
            """take sample that cover as much as possible the test units space"""
            return self.coverage_test_units_ucb_sampling(task_id)
        
    
    def refine(self, list_program_2_refine, list_task_id_repair, llm=None, tokenizer=None): 
        """refine program given and compute test units for new program"""
        # refine program + postprocess
        prompt_solve = prompt_repair_sol_v1 
        if self.crossover:
            prompt_solve = prompt_repair_cot_fewshot_crossover_v1
        # prompt formatting
        list_prompt=[]
        for id in range(len(list_program_2_refine)):
            prompt = get_repair_prompt(task2solve = self.data2test[list_task_id_repair[id]],
                                    list_previous_response = list_program_2_refine[id],
                                    prompt_solver = prompt_solve,
                                    grid_display_mode="numpy",
                                    alt_colors = True,
                                    prompt_colors = True,
                                    max_example = -1)
            list_prompt.append(prompt)
        
        # sglang, vllm or openai generation
        if self.sglang:
                results = llm.generate(list_prompt, n=self.n_completions)
                dict_response = format_all_generation(results, list_task_id_repair, use_vllm_generate=False, use_cot=False)

        elif not self.openai:
            from vllm import  SamplingParams

            prompts_formated_tokenized = prompt2token_chat(tokenizer, list_prompt)
            sampling_params = SamplingParams(temperature=self.temperature, max_tokens=self.max_tokens, stop_token_ids=[tokenizer.eos_token_id], n=self.n_completions, min_p=0.05)

            # generation
            results = llm.generate(prompt_token_ids=prompts_formated_tokenized, sampling_params=sampling_params)
            dict_response = format_all_generation(results, list_task_id_repair, use_vllm_generate=True,keep_full_tex=True)

        
        else: 
            if self.gpt_mini_mode:
                results = get_multiple_completions_multiple_client(self.client, list_prompt,n=self.n_completions,max_workers=self.max_workers)
            else:
                results = get_multiple_completions(self.client,list_prompt,cfg_generation=self.cfg_generation,n=self.n_completions,max_workers=self.max_workers)
            dict_response = format_all_generation(results, list_task_id_repair, use_vllm_generate=False,keep_full_tex=True)

        dict_response_refined = check_solutions(dict_response,self.data2test, keep_only_correct_results=False)
        return dict_response_refined

    def stop_condition(self,task_id):

        task_responses = self.archive[task_id]
        if self.split == "train":
            sum_correct_train_input = sum([all(response["correct_train_input"] + response["correct_test_input"])  for response in task_responses if response["type"] == "refined"])
        else:
            sum_correct_train_input = sum([all(response["correct_train_input"])  for response in task_responses if response["type"] == "refined"])

        return sum_correct_train_input >= 25 
    
    def run(self,llm=None, tokenizer=None,save=True):
        print(get_number_of_solved_tasks(self.archive))
        for iter in trange(self.N_budget):
            list_program_2_refine=[]
            list_task_id_repair = [] 
            for task_id in self.list_task_id:
                #TODO: stop condition
                stop_condition = self.stop_condition(task_id)
                if stop_condition:
                    continue
                list_program_2_refine_task_id = self.sample_program_2_refine(task_id)
                if isinstance(list_program_2_refine_task_id,np.ndarray):
                    list_program_2_refine_task_id = list_program_2_refine_task_id.tolist()
                list_program_2_refine.append(list_program_2_refine_task_id)
                list_task_id_repair.append(task_id)

            # assert len(list_program_2_refine) == len(self.list_task_id)

            dict_response_refined = self.refine(list_program_2_refine,list_task_id_repair, llm, tokenizer)
            list_program_2_refine 
            # update archive
            id=0
            for task_id,response in dict_response_refined.items():
                # add unique_id and task_id and N
                for resp in response:
                    if "correct_train_input" in resp:
                        # weird bug sometimes
                        resp["unique_id"] = str(self.unique_id)
                        self.unique_id += 1
                        resp["task_id"] = task_id
                        resp["N"] = 0
                        resp["parents"] = [i["unique_id"] for i in list_program_2_refine[id]]
                        resp["type"] = "refined"
                        
                        # if np.all(resp["correct_train_input"]) and np.all(resp["correct_test_input"]):
                            # print("found one correct solution; for task_id",task_id)
                            # print("found one correct solution;")
                            # print(resp["full_text"])
                        self.archive[task_id].append(resp)
                    # take into account the n completions 

                id+=1
            try:
                self.results = get_number_of_solved_tasks(self.archive)
                print(self.results)
            except:
                pass
            # if save:
            #     if (iter+1) % 2 == 0:
            #         self.save_archive()
            self.save_archive()
        # if save:
        #     self.save_archive()
        self.save_archive()