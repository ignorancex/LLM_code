import json
import torch
import time
import os
import re
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from typing import List, Dict

# Add distributed training related global variables
global_rank = None
global_world_size = None
global_local_rank = None

def setup_distributed():
    """Setup distributed training environment"""
    global global_rank, global_world_size, global_local_rank
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU.")
        return False
    
    # Get local rank from environment variable
    global_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if global_local_rank == -1:
        print("LOCAL_RANK environment variable not set. Using single GPU.")
        return False
    
    # Initialize process group
    dist.init_process_group(backend="nccl")
    
    # Get rank and world size
    global_rank = dist.get_rank()
    global_world_size = dist.get_world_size()
    
    # Set device for this process
    torch.cuda.set_device(global_local_rank)
    
    print(f"Initialized process {global_rank}/{global_world_size} on GPU {global_local_rank}")
    return True

class MultiDoctorMedicalSimulation:
    def __init__(self, model_path, input_file, output_file, temperature=0.7, top_p=0.9, device_map="auto", batch_size=8):
        """
        Initialize the multi-doctor medical dialogue simulation system
        Args:
            model_path (str): Path to the doctor model.
            input_file (str): Path to the input JSON file containing dialogue data.
            output_file (str): Path to the output JSON file to save simulation results.
            temperature (float): Temperature parameter for the doctor model generation.
            top_p (float): Top-p parameter for the doctor model generation.
            device_map (str): Device mapping for model loading.
            batch_size (int): Batch size for processing.
        """
        self.input_file = input_file
        self.output_file = output_file
        self.temperature = temperature
        self.top_p = top_p
        self.batch_size = batch_size
        self.num_doctors = 3  # Number of doctor agents
        self.doctor_names = [f"Doctor_{i+1}" for i in range(self.num_doctors)]

        # Set up distributed environment
        is_distributed = setup_distributed()
        if is_distributed:
            self.device = torch.device(f"cuda:{global_local_rank}")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Process {global_rank if is_distributed else 0}: Using device: {self.device}")

        # Load doctor model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
        # Ensure padding token is set for batch processing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # In distributed environment, each process loads model to its own GPU
        if is_distributed:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=None,  # Don't use device_map, manage manually
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            self.model.to(self.device)
        else:
            # Non-distributed environment, use device_map for automatic management
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device_map,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )

        # Load patient model and tokenizer (assuming patient model uses Qwen2.5-7B-Instruct)
        # Note: If patient model path is different, modify here
        patient_model_path = "Qwen2.5-7B-Instruct"
        self.patient_tokenizer = AutoTokenizer.from_pretrained(patient_model_path, trust_remote_code=True, padding_side='left')
        # Ensure padding token is set for batch processing
        if self.patient_tokenizer.pad_token is None:
            self.patient_tokenizer.pad_token = self.patient_tokenizer.eos_token
            self.patient_tokenizer.pad_token_id = self.patient_tokenizer.eos_token_id

        # In distributed environment, each process loads patient model to its own GPU
        if is_distributed:
            self.patient_model = AutoModelForCausalLM.from_pretrained(
                patient_model_path,
                device_map=None,  # Don't use device_map, manage manually
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            self.patient_model.to(self.device)
        else:
            # Non-distributed environment, use device_map for automatic management
            self.patient_model = AutoModelForCausalLM.from_pretrained(
                patient_model_path,
                device_map=device_map,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )

        # Doctor system prompts - one for each doctor (all use the same base template)
        self.doctor_system_prompt = """You are an experienced doctor who needs to provide professional diagnosis and recommendation to patients through consultation. Please listen carefully to the patient's description, ask targeted questions, and collect sufficient information before giving a diagnosis and treatment recommendation.

Quick Guide
Objectives:
1. Obtain key information through effective questioning, each round of questions should be modified based on the previous round's content, meaning you shouldn't ask similar questions.
2. Comprehensively analyze the patient's condition to provide an accurate diagnosis and appropriate treatment recommendations.

Rules:
1. You can only choose one of the options to respond, you cannot both answer questions and provide a diagnosis simultaneously.
2. Absolutely do not repeat or ask questions similar or identical to those previously asked.

Response:
<think> [your thinking] </think>
<answer>If you believe there is insufficient information, please only ask one question, in this format:
Question: (your question).
</answer> | <answer>If you believe you have obtained enough information, please only provide diagnosis and recommendations, in this format:
Diagnosis: (the patient's most likely disease or symptoms)
Recommendation: (corresponding treatment plan or recommendation)
</answer>
Rewards:
Incorrect format: -2.0
Effective question (patient can provide an answer and the question is helpful for diagnosis): +1.0
Ineffective questions do not count towards score
Repeated questions: -2.0
The number of conversation turn is limited. Reaching maximum interaction rounds without providing a diagnosis: -5.0
Completely correct diagnosis and recommendations: +10.0
      """

        # Doctor review system prompt - for doctors to review others' diagnoses
        self.doctor_review_system_prompt = """You are a professional doctor. You are making a diagnosis and providing recommendations for a patient whose self-report is as follows:
#Patient Self-Report#
{self_report}

Regarding the patient's condition, you have provided an initial diagnosis:
#Diagnosis Result#
{doctor_diagnosis}
#Recommendation Result#
{doctor_recommendation}

(1) Below, you will receive diagnostic opinions from other doctors, including their diagnosis results and recommendations. You need to critically review and analyze the other doctors' diagnostic opinions. Other doctors' results: {other_doctors_results}
(2) If you find parts of the other doctors' diagnostic opinions more reasonable than yours, please incorporate them to improve your diagnosis.
(3) If you believe your diagnostic opinion is more scientifically sound compared to others, please maintain your opinion unchanged.
(4) Please format your output as follows:

<answer>
Diagnosis: (the patient's most likely disease or symptoms)
Recommendation: (corresponding treatment plan or recommendation)
</answer>
"""

        # Moderator system prompt - for the final diagnosis
        self.moderator_system_prompt = """You are a senior Chief Physician.
You are hosting a medical consultation meeting for a patient. Here are the diagnoses and recommendations from multiple doctors:
{doctors_diagnoses_and_recommendations}

The patient's basic situation is as follows:
{self_report}

(1) You need to consider each doctor's diagnostic report, which contains Diagnosis Results and Recommendation Results for the patient.
(2) You need to summarize the information from each doctor and provide a final diagnosis for the patient.
(3) Please format your output as follows:

<answer>
Diagnosis: (the patient's most likely disease or symptoms)
Recommendation: (corresponding treatment plan or recommendation)
</answer>
"""

        # Final diagnosis system prompt
        self.final_diagnosis_system_prompt = """You are an experienced doctor who must provide a diagnosis and recommendation based on existing information. You have already asked enough questions and must now give the final diagnosis and treatment recommendation.

Based on the available information, please provide your best possible diagnosis and recommendation, even if the information is incomplete.

Respond strictly in the following format:
<think> [your thinking] </think>
<answer>
Diagnosis: (the most likely disease or condition)
Recommendation: (corresponding treatment plan or suggestion)
</answer>

Do not include any other content. Be concise.
"""

    def load_dialogue_data(self):
        """Load dialogue data"""
        with open(self.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def extract_doctor_questions_and_patient_responses(self, dialogue):
        """Extract doctor questions and corresponding patient responses from dialogue"""
        questions_responses = []

        for turn in dialogue:
            doctor_questions = turn.get("doctor_question", [])
            patient_responses = turn.get("patient_response", [])

            if not isinstance(doctor_questions, list):
                doctor_questions = [doctor_questions]
            if not isinstance(patient_responses, list):
                patient_responses = [patient_responses]

            if doctor_questions and patient_responses:
                questions_responses.append({
                    "doctor_questions": doctor_questions,
                    "patient_responses": patient_responses
                })

        return questions_responses

    def process_doctor_response(self, doctor_response):
        """Process doctor's response, determine if it's a question or diagnosis"""
        ori_doctor_response = doctor_response.strip()
        # Extract content within <answer> tags
        doctor_response = re.search(r"<answer>(.*?)</answer>", ori_doctor_response, re.DOTALL)
        if doctor_response:
            doctor_response = doctor_response.group(1).strip()
        elif "Question:" in ori_doctor_response:
            doctor_response = "Question: " + ori_doctor_response.split("Question:")[1].strip()
        else:
            # If no <answer> tag, treat the entire response as a question
            return {"type": "question", "content": ori_doctor_response}

        # Determine if it's a question or diagnosis
        if doctor_response.startswith("Question:") or doctor_response.startswith("Question："):
            match = re.search(r"Question[:：](.*?)(?=\n|$)", doctor_response, re.DOTALL)
            if match:
                question = match.group(1).strip()
                return {"type": "question", "content": question}
            else:
                return {"type": "question", "content": doctor_response}
        elif doctor_response.startswith("Diagnosis:") or doctor_response.startswith("Diagnosis："):
            diagnosis_match = re.search(r"Diagnosis[:：](.*?)(?=Recommendation[:：]|$)", doctor_response, re.DOTALL)
            recommend_match = re.search(r"Recommendation[:：](.*?)(?=\n|$)", doctor_response, re.DOTALL)

            diagnosis = diagnosis_match.group(1).strip() if diagnosis_match else ""
            recommendation = recommend_match.group(1).strip() if recommend_match else ""

            return {"type": "diagnosis", "diagnosis": diagnosis, "recommendation": recommendation}
        else:
            # If doctor is making a diagnosis
            if "Diagnosis" in doctor_response and "Recommendation" in doctor_response:
                diagnosis_match = re.search(r"Diagnosis[:：](.*?)(?=Recommendation[:：]|$)", doctor_response, re.DOTALL)
                recommend_match = re.search(r"Recommendation[:：](.*?)(?=\n|$)", doctor_response, re.DOTALL)

                diagnosis = diagnosis_match.group(1).strip() if diagnosis_match else ""
                recommendation = recommend_match.group(1).strip() if recommend_match else ""

                return {"type": "diagnosis", "diagnosis": diagnosis, "recommendation": recommendation}
            else:
                # Otherwise treat as question
                return {"type": "question", "content": doctor_response}

    def count_tokens(self, text):
        """Count the number of tokens in the text"""
        # Use the doctor model's tokenizer to count tokens
        return len(self.tokenizer.encode(text))

    def generate_final_diagnosis(self, self_report, dialogue_history_messages):
        """
        Generate final diagnosis and recommendation (for forced diagnosis at max turns)
        Args:
            self_report (str): Patient's initial self-report.
            dialogue_history_messages (list): List of dialogue history messages.
        Returns:
            dict: Result containing diagnosis and recommendation.
        """
        messages = [
            {"role": "system", "content": self.final_diagnosis_system_prompt},
            {"role": "user", "content": f"Patient self-report:\n{self_report}"}
        ]

        messages.extend(dialogue_history_messages)

        # Apply chat template and tokenize
        formatted_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        prompt_tokens = self.count_tokens(formatted_input)

        inputs = self.tokenizer(formatted_input, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id # Ensure pad token is handled correctly
            )

        # Decode response and extract diagnosis and recommendation
        response_tokens = outputs.sequences[0].size(0) - inputs.input_ids.size(1)
        response = self.tokenizer.decode(outputs.sequences[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

        diagnosis_match = re.search(r"Diagnosis[:：](.*?)(?=Recommendation[:：]|$)", response, re.DOTALL)
        recommend_match = re.search(r"Recommendation[:：](.*?)(?=\n|$)", response, re.DOTALL)

        diagnosis = diagnosis_match.group(1).strip() if diagnosis_match else "Unable to determine diagnosis"
        recommendation = recommend_match.group(1).strip() if recommend_match else "Recommend further examination"

        return {
            "type": "diagnosis",
            "diagnosis": diagnosis,
            "recommendation": recommendation,
            "tokens": response_tokens,
            "prompt_tokens": prompt_tokens
        }

    def generate_doctor_review(self, self_report, doctor_name, doctor_diagnosis, doctor_recommendation, other_doctors_results):
        """
        Generate doctor's revised diagnosis after seeing other doctors' diagnoses
        Args:
            self_report (str): Patient's self-report
            doctor_name (str): Current doctor's name
            doctor_diagnosis (str): Current doctor's initial diagnosis
            doctor_recommendation (str): Current doctor's initial recommendation
            other_doctors_results (str): Formatted string of other doctors' diagnoses and recommendations
        Returns:
            dict: Updated diagnosis and recommendation
        """
        # Format the prompt for doctor review
        review_prompt = self.doctor_review_system_prompt.format(
            self_report=self_report,
            doctor_diagnosis=doctor_diagnosis,
            doctor_recommendation=doctor_recommendation,
            other_doctors_results=other_doctors_results
        )
        
        messages = [
            {"role": "system", "content": review_prompt},
            {"role": "user", "content": "Please review the diagnoses and update your assessment if needed."}
        ]

        # Apply chat template and tokenize
        formatted_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        prompt_tokens = self.count_tokens(formatted_input)

        inputs = self.tokenizer(formatted_input, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # Decode response and extract updated diagnosis and recommendation
        response_tokens = outputs.sequences[0].size(0) - inputs.input_ids.size(1)
        response = self.tokenizer.decode(outputs.sequences[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

        # Process the response
        processed_response = self.process_doctor_response(response)
        
        if processed_response["type"] == "diagnosis":
            return {
                "type": "diagnosis",
                "diagnosis": processed_response["diagnosis"],
                "recommendation": processed_response["recommendation"],
                "tokens": response_tokens,
                "prompt_tokens": prompt_tokens
            }
        else:
            # If for some reason the response isn't a diagnosis, return the original
            return {
                "type": "diagnosis",
                "diagnosis": doctor_diagnosis,
                "recommendation": doctor_recommendation,
                "tokens": response_tokens,
                "prompt_tokens": prompt_tokens
            }

    def generate_moderator_summary(self, self_report, doctors_diagnoses):
        """
        Generate moderator's final summary and recommendation based on all doctors' diagnoses
        Args:
            self_report (str): Patient's self-report
            doctors_diagnoses (dict): Dictionary of doctors' diagnoses and recommendations
        Returns:
            dict: Final diagnosis and recommendation
        """
        import random
        
        # Format all doctors' diagnoses
        doctors_results_str = ""
        for doctor_name, diagnosis in doctors_diagnoses.items():
            doctors_results_str += f"{doctor_name}:\n"
            doctors_results_str += f"Diagnosis: {diagnosis['diagnosis']}\n"
            doctors_results_str += f"Recommendation: {diagnosis['recommendation']}\n\n"
        
        # Format the prompt for moderator
        moderator_prompt = self.moderator_system_prompt.format(
            doctors_diagnoses_and_recommendations=doctors_results_str,
            self_report=self_report
        )
        
        messages = [
            {"role": "system", "content": moderator_prompt},
            {"role": "user", "content": "Please provide the final diagnosis and recommendation based on all doctors' assessments."}
        ]

        # Apply chat template and tokenize
        formatted_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        prompt_tokens = self.count_tokens(formatted_input)

        inputs = self.tokenizer(formatted_input, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # Decode response and extract final diagnosis and recommendation
        response_tokens = outputs.sequences[0].size(0) - inputs.input_ids.size(1)
        response = self.tokenizer.decode(outputs.sequences[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

        # Process the response
        processed_response = self.process_doctor_response(response)
        
        # Initialize result with moderator's response
        result = {
            "type": "diagnosis",
            "diagnosis": "",
            "recommendation": "",
            "tokens": response_tokens,
            "prompt_tokens": prompt_tokens
        }
        
        # If moderator provided a diagnosis, use it
        if processed_response["type"] == "diagnosis":
            result["diagnosis"] = processed_response["diagnosis"]
            result["recommendation"] = processed_response["recommendation"]
        
        # Collect all non-empty diagnoses and recommendations from doctors
        all_diagnoses = []
        all_recommendations = []
        for doctor_name, diagnosis_data in doctors_diagnoses.items():
            if diagnosis_data.get("diagnosis") and diagnosis_data["diagnosis"].strip():
                all_diagnoses.append(diagnosis_data["diagnosis"])
            if diagnosis_data.get("recommendation") and diagnosis_data["recommendation"].strip():
                all_recommendations.append(diagnosis_data["recommendation"])
        
        # Ensure both diagnosis and recommendation are non-empty
        if not result["diagnosis"] and all_diagnoses:
            result["diagnosis"] = random.choice(all_diagnoses)
        
        if not result["recommendation"] and all_recommendations:
            result["recommendation"] = random.choice(all_recommendations)
        
        # Final fallback
        if not result["diagnosis"]:
            result["diagnosis"] = "Unable to determine diagnosis based on available information"
        
        if not result["recommendation"]:
            result["recommendation"] = "Recommend further examination and consultation with a specialist"
            
        return result

    def batch_generate_doctor_responses(self, dialogue_states, doctor_index):
        """
        Batch generate doctor responses (questions or diagnoses) for a specific doctor.
        Args:
            dialogue_states (list): List of dialogue states needing doctor responses.
                                   Each state contains: 'id', 'self_report', 'dialogue_history_messages', 'iteration', etc.
            doctor_index (int): Index of the doctor (0, 1, or 2)
        Returns:
            list: List of results for each dialogue containing doctor responses.
                 Each result contains: 'dialogue_id', 'type', 'content'/'diagnosis'/'recommendation', 'tokens', 'prompt_tokens'.
        """
        if not dialogue_states:
            return []

        doctor_name = self.doctor_names[doctor_index]
        
        # Prepare doctor model input prompts for batch processing
        batched_prompts = []
        for state in dialogue_states:
            # Build prompt for current round
            cur = f"""\nPatient's description: {state['self_report']}
            Decide next action:
            Always output: <think> [your thinking] </think> <answer> [your response] </answer> No additional text. Strictly follow this format.
            """
            messages = [
                {"role": "user", "content": self.doctor_system_prompt + cur}
            ]
            # Add dialogue history
            messages.extend(state['dialogue_history_messages'].get(doctor_name, []))
            # Apply chat template
            formatted_input = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            # Add <think> to guide model's thinking process
            formatted_input = formatted_input + "<think>"
            batched_prompts.append(formatted_input)

        # Tokenize and pad prompts for batch processing
        inputs = self.tokenizer(batched_prompts, return_tensors="pt", padding='longest', truncation=True).to(self.model.device)

        # Calculate token count for each prompt (before padding)
        doctor_prompt_tokens = [self.count_tokens(p) for p in batched_prompts]

        # Batch generate doctor responses
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256, # Control maximum token generation
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id # Ensure pad token is handled correctly
            )

        # Decode batch-generated responses
        # Slice to remove input prompt tokens
        batched_sequences = outputs.sequences
        decoded_responses = self.tokenizer.batch_decode(batched_sequences[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Process and return results for each dialogue
        processed_results = []
        for i, response in enumerate(decoded_responses):
            processed_response = self.process_doctor_response(response.strip())
            # Add token statistics
            processed_response["prompt_tokens"] = doctor_prompt_tokens[i]
            processed_response["tokens"] = batched_sequences[i].size(0) - inputs.input_ids.shape[1] # Number of generated tokens

            # Include original dialogue state information for later update
            processed_response["dialogue_id"] = dialogue_states[i]['id']
            processed_response["doctor_name"] = doctor_name
            processed_response["ori_response"] = "<think>" + response

            processed_results.append(processed_response)

        return processed_results

    def batch_generate_patient_responses(self, dialogue_states):
        """
        Batch generate patient responses.
        Args:
            dialogue_states (list): List of dialogue states needing patient responses.
                                   Each state contains: 'id', 'doctor_question', 'questions_responses', 'dialogue_history',
                                   and optional 'enhanced_description' and 'dialogue_history_messages'.
        Returns:
            list: List of results for each dialogue containing patient responses.
                 Each result contains: 'dialogue_id', 'content', 'original_state_index', 'doctor_name'.
        """
        if not dialogue_states:
            return []

        judge_prompts = []
        patient_prompts = []
        state_indices = []
        dialogue_ids = []
        descriptions = []
        conversation_histories = []
        doctor_names = []

        for index, state in enumerate(dialogue_states):
            dialogue_id = state['id']
            doctor_name = state['doctor_name']
            doctor_question = state['doctor_question']
            # Get conversation history for this doctor
            conversation_history = state.get('dialogue_history_messages', {}).get(doctor_name, [])[:-1]  # Exclude current doctor question
            description = state.get('enhanced_description', "")

            judge_prompt = self._prepare_judge_prompt(doctor_question, conversation_history)
            judge_prompts.append(judge_prompt)
            patient_prompts.append(self._prepare_patient_prompt(doctor_question, description))
            state_indices.append(index)
            dialogue_ids.append(dialogue_id)
            descriptions.append(description)
            conversation_histories.append(conversation_history)
            doctor_names.append(doctor_name)

        judge_responses = self._batch_generate(judge_prompts, max_new_tokens=128)
        patient_responses = self._batch_generate(patient_prompts, max_new_tokens=128)

        processed_results = []
        for i in range(len(dialogue_states)):
            judge_response_text = self._parse_judge_response(judge_responses[i])
            if "Sorry" in judge_response_text:
                patient_response_text = "Sorry, you've asked this question before."
            else:
                patient_response_text = patient_responses[i].strip()

            processed_results.append({
                "dialogue_id": dialogue_ids[i],
                "content": patient_response_text,
                "original_state_index": state_indices[i],
                "doctor_name": doctor_names[i]
            })

        return processed_results

    def _batch_generate(self, prompts: List[str], max_new_tokens: int) -> List[str]:
        """
        Batch generate text.

        Args:
            prompts (List[str]): List of input prompts.
            max_new_tokens (int): Maximum number of tokens to generate.

        Returns:
            List[str]: List of generated texts.
        """
        if not prompts:
            return []

        inputs = self.patient_tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.patient_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.01,
                top_p=0.9,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.patient_tokenizer.eos_token_id
            )
        batched_sequences = outputs.sequences
        responses = self.patient_tokenizer.batch_decode(batched_sequences[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return responses

    def _parse_judge_response(self, response: str) -> str:
        """
        Parse response judging if a question is a repeat.

        Args:
            response (str): Original response generated by the model.

        Returns:
            str: Parsed response ("OK" or "Sorry, you've asked this question before.").
        """

        match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return "OK" # Default to OK if parsing fails

    def _prepare_judge_prompt(self, doctor_question: str, conversation_history: List[Dict]) -> str:
        """
        Prepare a prompt for the environment LLM to judge if the question is a repeat.

        Args:
            doctor_question (str): The doctor's question
            conversation_history (List[Dict]): The conversation history

        Returns:
            str: A formatted prompt for the environment LLM
        """
        history_questions = []
        for conv in conversation_history:
            if conv['role'] == 'assistant':
                history_questions.append(conv.get('extract_response', ''))
        question = doctor_question
        if len(history_questions) == 0:
            system_prompt = f"""Just Output OK.
OUTPUT FORMAT: <think>[Your thoughts here]</think><answer>[Your response here]</answer>."""
            prompt = """Output OK."""
        else:
            system_prompt = f"""You are a patient interacting with a doctor. Instructions for Responding to Medical Questions:
Compare the doctor's current question with doctor's history questions, and determine whether the current question is a repeat of a previously asked question (has the similar meaning).
If it is a repeat, please respond with "Sorry, you've asked this question before."
If it is not a repeat, please respond with "OK."
OUTPUT FORMAT: <think>[Your thoughts here]</think><answer>[Your response here]</answer>."""
            prompt = f"""
Doctor's History questions:
[
{"".join(history_questions)}
]

Doctor's Current Question: {question}
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        prompt = self.patient_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        return prompt + "<think>"

    def _prepare_patient_prompt(self, doctor_question: str, description: str) -> str:
        """
        Prepare a prompt for the environment LLM to answer the doctor's question.

        Args:
            doctor_question (str): The doctor's question
            description (str): The patient's self-report states

        Returns:
            str: A formatted prompt for the environment LLM
        """
        system_prompt = f"""You are a patient interacting with a doctor. Instructions for Responding to Medical Questions:
Answer each medical question from the doctor concisely in a single sentence, strictly describing your symptoms and avoiding any mention of diagnoses and recommendations.
If the question is unrelated to your self-report states: "Sorry, I cannot answer your question."

Your self-report states: {description}
"""

        prompt = f"""{doctor_question}"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        prompt = self.patient_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        return prompt

    def run_simulation(self, max_iterations=10, start_idx=0, end_idx=None, batch_size=8):
        """
        Run medical dialogue simulation with multiple doctors, supporting batch processing.
        Args:
            max_iterations (int): Maximum dialogue turns per doctor.
            start_idx (int): Starting index for dialogue processing (based on original dataset).
            end_idx (int): Ending index for dialogue processing (not inclusive) (based on original dataset).
            batch_size (int): Batch size.
        Returns:
            list: List of simulation results for all dialogues.
        """
        self.batch_size = batch_size # Update batch size

        dialogue_data = self.load_dialogue_data()

        # Slice data to process based on start_idx and end_idx
        if end_idx is None:
            end_idx = len(dialogue_data)
        # Store original indices and corresponding data
        dialogue_data_subset = [(i, dialogue_data[i]) for i in range(start_idx, end_idx)]

        # In distributed environment, each process only handles a portion of the data
        if global_world_size and global_world_size > 1:
            # Calculate how much data each process handles
            per_rank = len(dialogue_data_subset) // global_world_size
            remainder = len(dialogue_data_subset) % global_world_size
            
            # Calculate start and end indices for the current process
            start_idx = global_rank * per_rank + min(global_rank, remainder)
            end_idx = start_idx + per_rank + (1 if global_rank < remainder else 0)
            
            # Only process data assigned to the current process
            dialogue_data_subset = dialogue_data_subset[start_idx:end_idx]
            print(f"Process {global_rank}: Processing {len(dialogue_data_subset)} items (indices {start_idx}-{end_idx-1})")

        # Initialize active dialogues state list
        active_dialogues = []
        for original_idx, item in dialogue_data_subset:
            self_report = item.get("self_report", "")
            dialogue = item.get("dialogue", [])
            true_diagnosis = item.get("diagnosis", "")
            true_recommendation = item.get("recommendation", "")
            enhanced_description = item.get("enhanced_description", "")

            if not self_report or not dialogue or not enhanced_description:
                print(f"Dialogue {original_idx} is missing self-report or conversation content. Skipping.")
                continue

            # Extract original Q&A pairs for patient model response lookup
            questions_responses = self.extract_doctor_questions_and_patient_responses(dialogue)

            # Initialize dialogue state
            active_dialogues.append({
                "id": original_idx, # Use original index as dialogue ID
                "self_report": self_report,
                "true_diagnosis": true_diagnosis,
                "true_recommendation": true_recommendation,
                "enhanced_description": enhanced_description,
                "simulation_dialogue": {}, # Store simulated dialogue turns by doctor
                "dialogue_history_messages": {}, # Store dialogue history for model input by doctor
                "iteration": {doctor: 0 for doctor in self.doctor_names}, # Track current dialogue turn count by doctor
                "is_completed": {doctor: False for doctor in self.doctor_names}, # Track if dialogue is completed by doctor
                "questions_responses": questions_responses, # Store original Q&A pairs
                "has_diagnosis": {doctor: False for doctor in self.doctor_names}, # Track if a doctor has provided diagnosis
                "doctor_diagnoses": {} # Store each doctor's final diagnosis
            })

            # Initialize simulation_dialogue and dialogue_history_messages for each doctor
            for doctor in self.doctor_names:
                active_dialogues[-1]["simulation_dialogue"][doctor] = []
                active_dialogues[-1]["dialogue_history_messages"][doctor] = []

        simulation_results = [] # Store final results for all completed dialogues

        print(f"Starting simulation for {len(active_dialogues)} dialogues with batch size {self.batch_size}")

        # Main simulation loop for individual doctor consultations
        for doctor_idx, doctor_name in enumerate(self.doctor_names):
            print(f"Starting consultations with {doctor_name}")
            
            # Continue as long as there are active dialogues for this doctor
            doctor_active_dialogues = [d for d in active_dialogues if not d["is_completed"][doctor_name]]
            
            while doctor_active_dialogues:
                doctor_turn_dialogues = [] # Dialogues needing doctor's next response
                patient_turn_dialogues = [] # Dialogues needing patient's next response
                finished_dialogues_indices = [] # Indices of completed dialogues

                # Categorize active dialogues into doctor turn or patient turn
                for i, state in enumerate(doctor_active_dialogues):
                    if state['is_completed'][doctor_name]:
                        finished_dialogues_indices.append(i)
                        continue # Skip completed dialogues

                    # Check if maximum turns reached
                    if state['iteration'][doctor_name] >= max_iterations:
                        # Force final diagnosis
                        final_diagnosis = self.generate_final_diagnosis(
                            state['self_report'], 
                            state['dialogue_history_messages'][doctor_name]
                        )
                        diagnosis = final_diagnosis.get("diagnosis", "")
                        recommendation = final_diagnosis.get("recommendation", "")

                        # Record final diagnosis
                        state['simulation_dialogue'][doctor_name].append({
                            "turn": state['iteration'][doctor_name] + 1,
                            "role": "doctor",
                            "content": f"Diagnosis: {diagnosis}\nRecommendation: {recommendation}",
                            "tokens": final_diagnosis.get("tokens", 0),
                            "prompt_tokens": final_diagnosis.get("prompt_tokens", 0),
                            "is_diagnosis": True,
                            "is_forced": True
                        })
                        state['is_completed'][doctor_name] = True
                        state['has_diagnosis'][doctor_name] = True
                        state['doctor_diagnoses'][doctor_name] = {
                            "diagnosis": diagnosis,
                            "recommendation": recommendation
                        }
                        finished_dialogues_indices.append(i)
                        continue

                    # Determine if next turn is doctor or patient
                    # If dialogue history is empty, or last message is from patient, next is doctor
                    if not state['dialogue_history_messages'][doctor_name] or state['dialogue_history_messages'][doctor_name][-1]["role"] == "user":
                        doctor_turn_dialogues.append(state)
                    # If last message is from doctor, next is patient
                    elif state['dialogue_history_messages'][doctor_name][-1]["role"] == "assistant":
                        # Extract doctor's question for patient model
                        last_doctor_message_content = state['dialogue_history_messages'][doctor_name][-1]["content"]
                        processed_last_doctor_message = self.process_doctor_response(last_doctor_message_content)

                        if processed_last_doctor_message["type"] == "question":
                            # Store doctor question in state for patient batch processing
                            state['doctor_question'] = processed_last_doctor_message["content"]
                            state['doctor_name'] = doctor_name
                            patient_turn_dialogues.append(state)
                        else:
                            # If doctor's last message is not a question (it's a diagnosis), dialogue is complete
                            diagnosis = processed_last_doctor_message.get("diagnosis", "")
                            recommendation = processed_last_doctor_message.get("recommendation", "")
                            state['has_diagnosis'][doctor_name] = True
                            state['doctor_diagnoses'][doctor_name] = {
                                "diagnosis": diagnosis,
                                "recommendation": recommendation
                            }
                            state['is_completed'][doctor_name] = True
                            finished_dialogues_indices.append(i)

                # Remove completed dialogues from doctor_active_dialogues
                # Remove from the end to avoid index issues
                for i in sorted(finished_dialogues_indices, reverse=True):
                    doctor_active_dialogues.pop(i)

                # Batch process dialogues needing doctor responses
                if doctor_turn_dialogues:
                    print(f"Processing {len(doctor_turn_dialogues)} {doctor_name} turns in batches...")
                    # Process in batches according to batch_size
                    for i in tqdm(range(0, len(doctor_turn_dialogues), self.batch_size), desc=f"{doctor_name} Batches", leave=False):
                        batch_states = doctor_turn_dialogues[i:i + self.batch_size]
                        # Call batch generate doctor responses function
                        doctor_results = self.batch_generate_doctor_responses(batch_states, doctor_idx)

                        # Update each dialogue state based on batch results
                        for result in doctor_results:
                            dialogue_id = result["dialogue_id"]
                            # Find corresponding active dialogue state by ID
                            state = next((d for d in active_dialogues if d['id'] == dialogue_id), None)
                            if state:
                                state['iteration'][doctor_name] += 1 # Increment turn after doctor response

                                if result["type"] == "question":
                                    doctor_question = result["content"]
                                    # Record doctor question in simulated dialogue
                                    state['simulation_dialogue'][doctor_name].append({
                                        "turn": state['iteration'][doctor_name],
                                        "role": "doctor",
                                        "content": doctor_question,
                                        "tokens": result.get("tokens", 0),
                                        "prompt_tokens": result.get("prompt_tokens", 0)
                                    })
                                    # Add doctor question to dialogue history for next round model input
                                    state['dialogue_history_messages'][doctor_name].append({
                                        "role": "assistant", 
                                        "content": result["ori_response"], 
                                        "extract_response": doctor_question
                                    })

                                else: # If it's a diagnosis
                                    diagnosis = result.get("diagnosis", "")
                                    recommendation = result.get("recommendation", "")
                                    # Record diagnosis in simulated dialogue
                                    state['simulation_dialogue'][doctor_name].append({
                                        "turn": state['iteration'][doctor_name],
                                        "role": "doctor",
                                        "content": f"Diagnosis: {diagnosis}\nRecommendation: {recommendation}",
                                        "tokens": result.get("tokens", 0),
                                        "prompt_tokens": result.get("prompt_tokens", 0),
                                        "is_diagnosis": True
                                    })
                                    state['has_diagnosis'][doctor_name] = True
                                    state['doctor_diagnoses'][doctor_name] = {
                                        "diagnosis": diagnosis,
                                        "recommendation": recommendation
                                    }
                                    state['is_completed'][doctor_name] = True

                # Batch process dialogues needing patient responses
                if patient_turn_dialogues:
                    print(f"Processing {len(patient_turn_dialogues)} patient responses for {doctor_name} in batches...")
                    # Re-collect dialogue states needing patient responses, as active_dialogues list may have changed
                    current_patient_turn_states = []
                    for state in doctor_active_dialogues:
                        # Check if dialogue is not completed, has history, and last message is from doctor
                        if (not state['is_completed'][doctor_name] and 
                            state['dialogue_history_messages'][doctor_name] and 
                            state['dialogue_history_messages'][doctor_name][-1]["role"] == "assistant"):
                            # Ensure doctor's last message is a question
                            processed_last_doctor_message = self.process_doctor_response(
                                state['dialogue_history_messages'][doctor_name][-1]["content"]
                            )
                            if processed_last_doctor_message["type"] == "question":
                                # Set doctor_question and doctor_name for batch processing
                                state['doctor_question'] = processed_last_doctor_message["content"]
                                state['doctor_name'] = doctor_name
                                current_patient_turn_states.append(state)

                    # Process in batches according to batch_size
                    for i in tqdm(range(0, len(current_patient_turn_states), self.batch_size), desc=f"Patient Batches for {doctor_name}", leave=False):
                        batch_states = current_patient_turn_states[i:i + self.batch_size]
                        # Call batch generate patient responses function
                        patient_results = self.batch_generate_patient_responses(batch_states)

                        # Update each dialogue state based on batch results
                        for result in patient_results:
                            dialogue_id = result["dialogue_id"]
                            # Find corresponding active dialogue state by ID
                            state = next((d for d in active_dialogues if d['id'] == dialogue_id), None)
                            if state and result["doctor_name"] == doctor_name:
                                patient_response_content = result["content"]
                                # Record patient response in simulated dialogue
                                state['simulation_dialogue'][doctor_name].append({
                                    "turn": state['iteration'][doctor_name],
                                    "role": "patient",
                                    "content": patient_response_content
                                })
                                # Add patient response to dialogue history for next round model input
                                state['dialogue_history_messages'][doctor_name].append({
                                    "role": "user", 
                                    "content": patient_response_content + f"\nTurn {state['iteration'][doctor_name] + 1}/{max_iterations}."
                                })

                # Update active dialogues list for this doctor
                doctor_active_dialogues = [d for d in active_dialogues if not d["is_completed"][doctor_name]]

                # If no progress made in this iteration, break loop and force diagnosis for remaining dialogues
                if not doctor_turn_dialogues and not patient_turn_dialogues and doctor_active_dialogues:
                    print(f"Warning: No progress made in this iteration with {doctor_name}, forcing final diagnosis.")
                    # Force final diagnosis for remaining active dialogues
                    for state in doctor_active_dialogues:
                        if not state['is_completed'][doctor_name]:
                            final_diagnosis = self.generate_final_diagnosis(
                                state['self_report'], 
                                state['dialogue_history_messages'][doctor_name]
                            )
                            diagnosis = final_diagnosis.get("diagnosis", "")
                            recommendation = final_diagnosis.get("recommendation", "")

                            state['simulation_dialogue'][doctor_name].append({
                                "turn": state['iteration'][doctor_name] + 1,
                                "role": "doctor",
                                "content": f"Diagnosis: {diagnosis}\nRecommendation: {recommendation}",
                                "tokens": final_diagnosis.get("tokens", 0),
                                "prompt_tokens": final_diagnosis.get("prompt_tokens", 0),
                                "is_diagnosis": True,
                                "is_forced": True
                            })
                            state['is_completed'][doctor_name] = True
                            state['has_diagnosis'][doctor_name] = True
                            state['doctor_diagnoses'][doctor_name] = {
                                "diagnosis": diagnosis,
                                "recommendation": recommendation
                            }
                    doctor_active_dialogues = []

        # Doctors collaboration phase - review and revise diagnoses
        print("Starting doctors collaboration phase...")
        for state in active_dialogues:
            for doctor_name in self.doctor_names:
                # Skip doctors who didn't complete their diagnosis
                if not state['has_diagnosis'][doctor_name]:
                    continue
                
                # Format all doctors' diagnoses
            other_doctors_results = ""
            for other_doctor in self.doctor_names:
                if other_doctor != doctor_name and state['has_diagnosis'][other_doctor]:
                    other_diagnosis = state['doctor_diagnoses'][other_doctor].get("diagnosis", "")
                    other_recommendation = state['doctor_diagnoses'][other_doctor].get("recommendation", "")
                    other_doctors_results += f"{other_doctor}:\nDiagnosis: {other_diagnosis}\nRecommendation: {other_recommendation}\n\n"
                
                # If there are other diagnoses to review
                if other_doctors_results:
                    # Generate revised diagnosis after reviewing others
                    revised_diagnosis = self.generate_doctor_review(
                        state['self_report'],
                        doctor_name,
                        state['doctor_diagnoses'][doctor_name]["diagnosis"],
                        state['doctor_diagnoses'][doctor_name]["recommendation"],
                        other_doctors_results
                    )
                    
                    # Update doctor's diagnosis with revised version
                    state['doctor_diagnoses'][doctor_name] = {
                        "diagnosis": revised_diagnosis.get("diagnosis", state['doctor_diagnoses'][doctor_name]["diagnosis"]),
                        "recommendation": revised_diagnosis.get("recommendation", state['doctor_diagnoses'][doctor_name]["recommendation"]),
                    }
                    
                    # Record revision in simulated dialogue
                    state['simulation_dialogue'][doctor_name].append({
                        "turn": state['iteration'][doctor_name] + 1,
                        "role": "doctor",
                        "content": f"Revised Diagnosis: {revised_diagnosis.get('diagnosis')}\nRevised Recommendation: {revised_diagnosis.get('recommendation')}",
                        "tokens": revised_diagnosis.get("tokens", 0),
                        "prompt_tokens": revised_diagnosis.get("prompt_tokens", 0),
                        "is_revision": True
                    })

        # Moderator phase - generate final consensus diagnosis
        print("Starting moderator consensus phase...")
        for state in active_dialogues:
            # Generate moderator's final diagnosis based on all doctors' inputs
            moderator_diagnosis = self.generate_moderator_summary(
                state['self_report'],
                state['doctor_diagnoses']
            )
            
            # Record moderator's final diagnosis
            state['final_diagnosis'] = {
                "diagnosis": moderator_diagnosis.get("diagnosis", ""),
                "recommendation": moderator_diagnosis.get("recommendation", ""),
            }
            
            # Record final summary in simulation results
            simulation_results.append({
                "id": state['id'],
                "self_report": state['self_report'],
                "true_diagnosis": state['true_diagnosis'],
                "true_recommendation": state['true_recommendation'],
                "doctors_diagnoses": state['doctor_diagnoses'],
                "final_diagnosis": state['final_diagnosis'],
                "simulation_dialogue": state['simulation_dialogue'],
                "total_turns": {doctor: state['iteration'][doctor] for doctor in self.doctor_names},
                "enhanced_description": state['enhanced_description']
            })

        # In distributed environment, collect results from all processes
        if global_world_size and global_world_size > 1:
            # Gather results from all processes
            all_results = [None] * global_world_size
            dist.all_gather_object(all_results, simulation_results)
            
            # Only merge results on main process (rank 0)
            if global_rank == 0:
                # Flatten results from all processes
                simulation_results = [result for process_results in all_results for result in process_results]
            else:
                # Non-main processes return empty list
                simulation_results = []

        # Save final results to file
        if global_rank is None or global_rank == 0:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(simulation_results, f, ensure_ascii=False, indent=2)

            print(f"Simulation complete. Results saved to: {self.output_file}")

        return simulation_results


def main():
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Multi-Doctor Medical Dialogue Simulation with Qwen 2.5 (Batched)")
    parser.add_argument("--model_path", type=str, default="/play/fengyichun/Jiawei/Models/Qwen2.5-7B-Instruct",
                        help="Path to the Qwen model")
    parser.add_argument("--input_file", type=str, default="ragen/env/medical_consultation/evaluation/test.json",
                        help="Input JSON file containing dialogue data")
    parser.add_argument("--output_dir", type=str,
                        default="outputs/Qwen2.5-7B-Instruct/",
                        help="Output directory for the simulation results")
    parser.add_argument("--output_prefix", type=str, default="multi_doctor_test_output_batched",
                        help="Prefix for the output filename")
    parser.add_argument("--add_timestamp", action="store_true",
                        help="Add timestamp to output filename")
    parser.add_argument("--max_iterations", type=int, default=10,
                        help="Maximum number of dialogue turns per doctor")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for doctor model generation")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling parameter for doctor model")
    parser.add_argument("--device_map", type=str, default="auto",
                        help="Device map for model loading (auto, cuda:0, etc.)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed progress information")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Starting index for dialogue processing")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="Ending index for dialogue processing (None means process to the end)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for model inference")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training")

    # Add usage examples
    parser.epilog = """
    Example usage:
    python MultiDoctorMedicalSimulation.py --model_path /path/to/model --input_file input.json --output_dir ./results --add_timestamp --batch_size 16

    To process only a subset of dialogues:
    python MultiDoctorMedicalSimulation.py --start_idx 10 --end_idx 20 --add_timestamp --batch_size 8
    """
    parser.formatter_class = argparse.RawDescriptionHelpFormatter

    args = parser.parse_args()

    if args.verbose:
        print(f"Starting Multi-Doctor Medical Dialogue Simulation with Qwen 2.5 (Batched)")
        print(f"Model path: {args.model_path}")
        print(f"Input file: {args.input_file}")
        print(f"Temperature: {args.temperature}, Top-p: {args.top_p}")
        print(f"Maximum iterations per doctor: {args.max_iterations}")
        print(f"Batch size: {args.batch_size}")
        if args.start_idx > 0 or args.end_idx is not None:
            print(f"Processing dialogues from index {args.start_idx} to {args.end_idx if args.end_idx is not None else 'end'}")

    # Create output filename
    output_filename = args.output_prefix

    # If index range specified, add to filename
    if args.start_idx > 0 or args.end_idx is not None:
        range_info = f"_{args.start_idx}_to_{args.end_idx if args.end_idx is not None else 'end'}"
        output_filename += range_info

    # Add timestamp if requested
    if args.add_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename += f"_{timestamp}"

    output_filename += ".json"

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, output_filename)

    if args.verbose:
        print(f"Output will be saved to: {output_file}")
        print("Loading model and tokenizer...")

    start_time = time.time()

    # Create and run simulator
    simulator = MultiDoctorMedicalSimulation(
        model_path=args.model_path,
        input_file=args.input_file,
        output_file=output_file,
        temperature=args.temperature,
        top_p=args.top_p,
        device_map=args.device_map,
        batch_size=args.batch_size
    )

    results = simulator.run_simulation(
        max_iterations=args.max_iterations,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        batch_size=args.batch_size
    )

    # Calculate and report statistics
    end_time = time.time()
    total_time = end_time - start_time

    if args.verbose:
        print(f"\nSimulation complete in {total_time:.2f} seconds")
        print(f"Results saved to: {output_file}")

        # Calculate some basic statistics
        if results:
            total_dialogues = len(results)
            # Calculate average turns across all doctors
            doctor_names = [f"Doctor_{i+1}" for i in range(3)]
            avg_turns = {doctor: sum(r.get('total_turns', {}).get(doctor, 0) for r in results) / total_dialogues 
                         for doctor in doctor_names}

            print(f"\nStatistics:")
            print(f"- Total dialogues processed: {total_dialogues}")
            for doctor in doctor_names:
                print(f"- Average turns for {doctor}: {avg_turns[doctor]:.2f}")

    # In distributed environment, ensure all processes are complete
    if global_world_size and global_world_size > 1:
        dist.barrier()
        print(f"Process {global_rank}: Simulation completed")


if __name__ == "__main__":
    main()