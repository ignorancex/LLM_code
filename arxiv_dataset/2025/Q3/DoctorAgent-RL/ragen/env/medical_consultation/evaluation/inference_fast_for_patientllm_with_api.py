import json
import torch
import time
import os
import re
import argparse
# We will not use AutoModelForCausalLM and AutoTokenizer for the doctor model
# from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from typing import List, Dict

# Import OpenAI for DeepSeek-v3 API calls
from openai import OpenAI, AzureOpenAI
from openai import APIError, APIConnectionError, RateLimitError # Import specific exceptions

# Import AutoModelForCausalLM and AutoTokenizer for the patient model
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加分布式训练相关的全局变量
global_rank = None
global_world_size = None
global_local_rank = None

def setup_distributed():
    """Setup distributed training environment"""
    global global_rank, global_world_size, global_local_rank

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU.")
        # In a distributed setting, if CUDA is not available, we cannot proceed with nccl backend
        # For this script, we'll just return False and run in non-distributed mode
        return False

    # Get local rank from environment variable
    # LOCAL_RANK is typically set by the distributed launcher (e.g., torch.distributed.launch)
    global_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if global_local_rank == -1:
        # If LOCAL_RANK is not set, assume single GPU or CPU execution
        print("LOCAL_RANK environment variable not set. Using single GPU or CPU.")
        return False

    # Initialize process group
    # 'nccl' backend is recommended for GPU
    try:
        dist.init_process_group(backend="nccl")
    except Exception as e:
        print(f"Failed to initialize distributed process group: {e}")
        print("Falling back to non-distributed execution.")
        return False


    # Get rank and world size
    global_rank = dist.get_rank()
    global_world_size = dist.get_world_size()

    # Set device for this process
    torch.cuda.set_device(global_local_rank)

    print(f"Initialized process {global_rank}/{global_world_size} on GPU {global_local_rank}")
    return True

class MedicalDialogueSimulation:
    def __init__(self, doctor_model_name, input_file, output_file, temperature=0.7, top_p=0.9, patient_model_path="Qwen2.5-7B-Instruct", device_map="auto", batch_size=8):
        """
        初始化医患对话模拟系统
        Args:
            doctor_model_name (str): 医生模型名称 (DeepSeek-v3 model name).
            input_file (str): 包含对话数据的输入 JSON 文件路径.
            output_file (str): 保存模拟结果的输出 JSON 文件路径.
            temperature (float): 医生模型生成时的温度参数.
            top_p (float): 医生模型生成时的 top-p 参数.
            patient_model_path (str): 患者模型的路径.
            device_map (str): 患者模型加载的设备映射.
            batch_size (int): 批处理大小 (主要影响患者模型).
        """
        self.input_file = input_file
        self.output_file = output_file
        self.temperature = temperature
        self.top_p = top_p
        self.batch_size = batch_size # 存储批处理大小
        self.doctor_model_name = doctor_model_name # DeepSeek-v3 model name
        self.max_api_retries = 3 # Maximum retry attempts for API calls

        # 设置分布式环境
        is_distributed = setup_distributed()
        if is_distributed:
            self.device = torch.device(f"cuda:{global_local_rank}")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Process {global_rank if is_distributed else 0}: Using device: {self.device}")

        # 初始化 DeepSeek-v3 API 客户端
        # 从环境变量中获取您的 API Key
        api_key = os.environ.get("API_KEY")
        if not api_key:
            raise ValueError("API_KEY environment variable not set.")
        self.doctor_model_name = doctor_model_name
        if "gpt" in doctor_model_name.lower():
            base_url = "https://search.bytedance.net/gpt/openapi/online/v2/crawl"
            api_version = "2024-08-06"
            self.doctor_client = AzureOpenAI(
                azure_endpoint=base_url,
                api_version=api_version,
                api_key=api_key,
            )
        elif "qwen" in doctor_model_name.lower():
            self.doctor_client = OpenAI(
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                api_key=api_key,
            )
        elif "llama" in doctor_model_name.lower():
            self.doctor_client = OpenAI(
                base_url = "https://integrate.api.nvidia.com/v1",
                api_key = api_key
            )
        else:
            self.doctor_client = OpenAI(
                base_url="https://ark.cn-beijing.volces.com/api/v3",
                api_key=api_key,
            )
        print(f"Initialized API client for model: {self.doctor_model_name}")

        # 加载患者模型和分词器 (假设患者模型使用 Qwen2.5-7B-Instruct 或其他本地模型)
        # 设置 padding_side='left' for decoder-only models
        self.patient_tokenizer = AutoTokenizer.from_pretrained(patient_model_path, trust_remote_code=True, padding_side='left')
        # 确保 padding token 已设置，用于批处理
        if self.patient_tokenizer.pad_token is None:
            self.patient_tokenizer.pad_token = self.patient_tokenizer.eos_token
            self.patient_tokenizer.pad_token_id = self.patient_tokenizer.eos_token_id

        # 在分布式环境中，每个进程加载患者模型到自己的GPU
        if is_distributed:
            self.patient_model = AutoModelForCausalLM.from_pretrained(
                patient_model_path,
                device_map=None,  # 不使用device_map，手动管理
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            self.patient_model.to(self.device)
        else:
            # 非分布式环境，使用device_map自动管理
            self.patient_model = AutoModelForCausalLM.from_pretrained(
                patient_model_path,
                device_map=device_map,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        print(f"Loaded patient model from: {patient_model_path}")


        # 医生系统提示词
        self.doctor_system_prompt = """
You are an experienced doctor who needs to provide professional diagnosis and advice to patients through consultation. Please listen carefully to the patient's description, ask targeted questions, and collect sufficient information before giving a diagnosis and treatment recommendation.

Quick Guide
Objectives:
1. Obtain key information through effective questioning, each round of questions should be modified based on the previous round's content, meaning you shouldn't ask similar questions.
2. Comprehensively analyze the patient's condition to provide an accurate diagnosis and appropriate treatment recommendations.

Rules:
1. You can only choose one of the options to respond, you cannot both ask a question and provide a diagnosis simultaneously.
2. Absolutely do not repeat or ask questions similar or identical to those previously asked.

Rewards:
Incorrect format: -2.0
Effective question (patient can provide an answer and the question is helpful for diagnosis): +1.0
Ineffective questions do not count towards score
Repeated questions: -2.0
The number of conversation turn is limited. Reaching maximum interaction rounds without providing a diagnosis: -5.0
Completely correct diagnosis and recommendations: +10.0

Response Format:
<answer>If you believe there is insufficient information, please only ask **one question**, in this format:
Question: (your question).
</answer> | <answer>If you believe you have obtained enough information, please only provide diagnosis and recommendations, in this format:
Diagnosis: (the patient's most likely disease or symptoms)
Recommendation: (corresponding treatment plan or advice)
</answer>
Always output <answer>[Your answer]</answer>.
"""

        # 最终诊断提示词
        self.final_diagnosis_system_prompt = """You are an experienced doctor who must provide a diagnosis and recommendation based on existing information. You have already asked enough questions and must now give the final diagnosis and treatment advice.

Based on the available information, please provide your best possible diagnosis and recommendation, even if the information is incomplete.

Respond strictly in the following format:
<answer>
Diagnosis: (the most likely disease or condition)
Recommendation: (corresponding treatment plan or suggestion)
</answer>

Do not include any other content. Be concise.
"""

    def load_dialogue_data(self):
        """加载对话数据"""
        with open(self.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def extract_doctor_questions_and_patient_responses(self, dialogue):
        """从对话中提取医生问题和对应的患者回答"""
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
        """处理医生的回应，判断是继续提问还是给出诊断"""
        ori_doctor_response = doctor_response.strip()
        # 提取 <answer> 标签内的内容
        doctor_response = re.search(r"<answer>(.*?)</answer>", ori_doctor_response, re.DOTALL)
        if doctor_response:
            doctor_response = doctor_response.group(1).strip()
        elif "Question:" in ori_doctor_response:
            doctor_response = "Question: " + ori_doctor_response.split("Question:")[1].strip()
        elif "Diagnosis:" in ori_doctor_response:
            doctor_response = "Diagnosis: " + ori_doctor_response.split("Diagnosis:")[1].strip()
        else:
            # 如果没有 <answer> 标签，则将整个响应视为问题
            return {"type": "question", "content": ori_doctor_response}

        # 判断是问题还是诊断
        if doctor_response.startswith("Question:") or doctor_response.startswith("Question："):
            match = re.search(r"Question[:：](.*?)(?=\n|$)", doctor_response, re.DOTALL)
            if match:
                question = match.group(1).strip()
                return {"type": "question", "content": question}
            else:
                return {"type": "question", "content": doctor_response}
        elif doctor_response.startswith("Diagnosis:") or doctor_response.startswith("Diagnosis："):
            diagnosis_match = re.search(r"Diagnosis[:：](.*?)(?=Recommendation[:：]|$)", doctor_response, re.DOTALL)
            # advice_match = re.search(r"Recommendation[:：](.*?)(?=\n|$)", doctor_response, re.DOTALL)
            advice_match = re.search(r"Recommendation[:：](.*)", doctor_response, re.DOTALL)

            diagnosis = diagnosis_match.group(1).strip() if diagnosis_match else ""
            advice = advice_match.group(1).strip() if advice_match else ""

            return {"type": "diagnosis", "diagnosis": diagnosis, "advice": advice}
        else:
            # 如果不以 Question: 或 Diagnosis: 开头，但包含 Diagnosis 和 Recommendation，则视为诊断
            if "Diagnosis" in doctor_response and "Recommendation" in doctor_response:
                diagnosis_match = re.search(r"Diagnosis[:：](.*?)(?=Recommendation[:：]|$)", doctor_response, re.DOTALL)
                advice_match = re.search(r"Recommendation[:：](.*?)(?=\n|$)", doctor_response, re.DOTALL)

                diagnosis = diagnosis_match.group(1).strip() if diagnosis_match else ""
                advice = advice_match.group(1).strip() if advice_match else ""

                return {"type": "diagnosis", "diagnosis": diagnosis, "advice": advice}
            else:
                # 否则视为问题
                return {"type": "question", "content": doctor_response}


    def count_tokens(self, text):
        """计算文本的token数量 (使用患者模型的分词器作为近似)"""
        # Note: This is an approximation as we are using the patient tokenizer
        # DeepSeek API provides token usage, which is more accurate.
        return len(self.patient_tokenizer.encode(text))

    def generate_final_diagnosis(self, self_report, dialogue_history_messages):
        """
        生成最终诊断和建议 (用于达到最大轮次时强制诊断)
        Args:
            self_report (str): 患者的初始自述.
            dialogue_history_messages (list): 对话历史消息列表.
        Returns:
            dict: 包含诊断和建议的结果.
        """
        messages = [
            {"role": "system", "content": self.final_diagnosis_system_prompt},
            {"role": "user", "content": f"Patient self-report:\n{self_report}"}
        ]

        messages.extend(dialogue_history_messages)

        response = "Error generating diagnosis after multiple retries."
        prompt_tokens = self.count_tokens(json.dumps(messages)) # Approximate token count on error
        response_tokens = 0

        for attempt in range(self.max_api_retries):
            try:
                # 调用 DeepSeek-v3 API 生成最终诊断
                completion = self.doctor_client.chat.completions.create(
                    model=self.doctor_model_name,
                    messages=messages,
                    # temperature=self.temperature,
                    # top_p=self.top_p,
                    max_tokens=256 # Control generated token count
                )
                response = completion.choices[0].message.content.strip()
                prompt_tokens = completion.usage.prompt_tokens
                response_tokens = completion.usage.completion_tokens
                break # Break out of retry loop on success

            except (APIError, APIConnectionError, RateLimitError) as e:
                print(f"Attempt {attempt + 1}/{self.max_api_retries} failed for final diagnosis: {e}")
                if attempt < self.max_api_retries - 1:
                    time.sleep(2 ** attempt) # Exponential backoff
                else:
                    print(f"Max retries reached for final diagnosis.")
            except Exception as e:
                 print(f"An unexpected error occurred during final diagnosis attempt {attempt + 1}/{self.max_api_retries}: {e}")
                 if attempt < self.max_api_retries - 1:
                    time.sleep(2 ** attempt) # Exponential backoff
                 else:
                    print(f"Max retries reached for final diagnosis.")


        # 解码响应并提取诊断和建议
        diagnosis_match = re.search(r"Diagnosis[:：](.*?)(?=Recommendation[:：]|$)", response, re.DOTALL)
        advice_match = re.search(r"Recommendation[:：](.*?)(?=\n|$)", response, re.DOTALL)

        diagnosis = diagnosis_match.group(1).strip() if diagnosis_match else "Unable to determine diagnosis"
        advice = advice_match.group(1).strip() if advice_match else "Recommend further examination"

        return {
            "type": "diagnosis",
            "diagnosis": diagnosis,
            "advice": advice,
            "tokens": response_tokens,
            "prompt_tokens": prompt_tokens
        }

    def batch_generate_doctor_responses(self, dialogue_states):
        """
        批量生成医生的回复 (问题或诊断)。
        Args:
            dialogue_states (list): 需要生成医生回复的对话状态列表。
                                    每个状态包含: 'id', 'self_report', 'dialogue_history_messages', 'iteration' 等。
        Returns:
            list: 包含每个对话医生回复结果的列表。
                  每个结果包含: 'dialogue_id', 'type', 'content'/'diagnosis'/'advice', 'tokens', 'prompt_tokens'。
        """
        if not dialogue_states:
            return []

        processed_results = []
        # Iterate through each dialogue state in the batch and call the API individually
        for state in dialogue_states:
            # 构建当前轮次的 prompt
            cur = f"""\nPatient's description: {state['self_report']}"""
            messages = [
                # {"role": "system", "content": self.doctor_system_prompt}, # System prompt is included in the first user message as per original logic
                {"role": "user", "content": self.doctor_system_prompt + cur}
            ]
            # 添加对话历史
            messages.extend(state['dialogue_history_messages'])

            response = "Error generating response after multiple retries."
            prompt_tokens = self.count_tokens(json.dumps(messages)) # Approximate token count on error
            response_tokens = 0
            ori_response = response # Default original response in case of failure

            for attempt in range(self.max_api_retries):
                try:
                    completion = self.doctor_client.chat.completions.create(
                        model=self.doctor_model_name,
                        messages=messages,
                        # temperature=self.temperature,
                        # top_p=self.top_p,
                        max_tokens=256 # Control generated token count
                    )
                    response = completion.choices[0].message.content.strip()
                    prompt_tokens = completion.usage.prompt_tokens
                    response_tokens = completion.usage.completion_tokens
                    ori_response = response # Update original response on success
                    break # Break out of retry loop on success

                except (APIError, APIConnectionError, RateLimitError) as e:
                    print(f"Attempt {attempt + 1}/{self.max_api_retries} failed for dialogue {state['id']}: {e}")
                    if attempt < self.max_api_retries - 1:
                        time.sleep(2 ** attempt) # Exponential backoff
                    else:
                        print(f"Max retries reached for dialogue {state['id']}.")
                except Exception as e:
                    print(f"An unexpected error occurred during dialogue {state['id']} attempt {attempt + 1}/{self.max_api_retries}: {e}")
                    if attempt < self.max_api_retries - 1:
                        time.sleep(2 ** attempt) # Exponential backoff
                    else:
                        print(f"Max retries reached for dialogue {state['id']}.")


            processed_response = self.process_doctor_response(response)
            # 添加 token 统计信息
            processed_response["prompt_tokens"] = prompt_tokens
            processed_response["tokens"] = response_tokens

            # 包含原始对话的状态信息，以便后续更新
            processed_response["dialogue_id"] = state['id']
            processed_response["ori_response"] = ori_response # Use updated ori_response

            processed_results.append(processed_response)

        return processed_results

    def batch_generate_patient_responses(self, dialogue_states):
        """
        批量生成患者的回复。
        Args:
            dialogue_states (list): 需要生成患者回复的对话状态列表。
                                    每个状态包含: 'id', 'doctor_question', 'questions_responses' (原始数据集的问答对), 'conversation_history' (当前对话历史),
                                    以及可选的 'enhanced_description' and 'dialogue_history_messages'.
        Returns:
            list: 包含每个对话患者回复结果的列表。
                  每个结果包含: 'dialogue_id', 'content', 'original_state_index'。
        """
        if not dialogue_states:
            return []

        judge_prompts = []
        patient_prompts = []
        state_indices = []
        dialogue_ids = []
        descriptions = []
        conversation_histories = []

        for index, state in enumerate(dialogue_states):
            dialogue_id = state['id']
            doctor_question = state['doctor_question']
            conversation_history = state.get('dialogue_history_messages', [])[:-1] # Exclude current doctor question

            description = state.get('enhanced_description', "")

            # Prepare prompts for the patient model (local model)
            judge_prompt = self._prepare_judge_prompt(doctor_question, conversation_history)
            judge_prompts.append(judge_prompt)
            patient_prompts.append(self._prepare_patient_prompt(doctor_question, description))
            state_indices.append(index)
            dialogue_ids.append(dialogue_id)
            descriptions.append(description)
            conversation_histories.append(conversation_history)

        # Batch generate responses from the patient model
        judge_responses = self._batch_generate(judge_prompts, max_new_tokens=128, model_type="patient")
        patient_responses = self._batch_generate(patient_prompts, max_new_tokens=128, model_type="patient")

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
                "original_state_index": state_indices[i]
            })

        return processed_results

    def _batch_generate(self, prompts: List[str], max_new_tokens: int, model_type: str) -> List[str]:
        """
        批量生成文本 (使用患者模型)。

        Args:
            prompts (List[str]): 输入的提示列表。
            max_new_tokens (int): 生成的最大 token 数。
            model_type (str): 模型类型 ('patient').

        Returns:
            List[str]: 生成的文本列表。
        """
        if not prompts:
            return []

        if model_type == "patient":
            tokenizer = self.patient_tokenizer
            model = self.patient_model
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.01, # Patient model uses fixed lower temperature for less variability
                top_p=0.9,
                do_sample=False, # Patient model does not use sampling
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.eos_token_id
            )
        batched_sequences = outputs.sequences
        # Decode responses, removing the input prompt part
        responses = tokenizer.batch_decode(batched_sequences[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return responses

    def _parse_judge_response(self, response: str) -> str:
        """
        解析判断重复问题的回复。

        Args:
            response (str): 模型生成的原始回复。

        Returns:
            str: 解析后的回复 ("OK" 或 "Sorry, you've asked this question before.").
        """

        match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return "OK" # Default to OK if parsing fails

    def _prepare_judge_prompt(self, doctor_question: str, conversation_history: List[Dict]) -> str:
        """
        Prepare a prompt for the environment LLM (patient model) to judge if the question is a repeat.

        Args:
            doctor_question (str): The doctor's question
            conversation_history (List[Dict]): The conversation history (messages in OpenAI format)

        Returns:
            str: A formatted prompt for the patient model
        """
        history_questions = []
        # Extract doctor questions from the history
        for conv in conversation_history:
            if conv['role'] == 'assistant':
                # Assuming 'content' for assistant role contains the doctor's response
                # We need to process this content to extract the question part
                processed_response = self.process_doctor_response(conv['content'])
                if processed_response['type'] == 'question':
                    history_questions.append(processed_response['content'])

        question = doctor_question
        if len(history_questions) == 0:
            # If no history, it cannot be a repeat
            system_prompt = f"""Just Output OK.
OUTPUT FORMAT: <think>[Your thoughts here]</think><answer>[Your response here]</answer>."""
            prompt = """Output OK."""
        else:
            system_prompt = f"""You are a patient interacting with a doctor. Instructions for Responding to Medical Questions:
Compare the doctor's current question with doctor's history questions, and determine whether the current question is a repeat of a previously asked question (has the similar meaning).
If it is a repeat, please respond with "Sorry, you've asked this question before."
If it is not a repeat, please respond with "OK."
OUTPUT FORMAT: <think>[Your thoughts here]</think><answer>[Your response here]</answer>."""
            # Format history questions for the prompt
            history_questions_str = "\n".join([f"- {q}" for q in history_questions])
            prompt = f"""
Doctor's History questions:
{history_questions_str}

Doctor's Current Question: {question}
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        # Apply chat template using patient tokenizer
        prompt_text = self.patient_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        return prompt_text + "<think>" # Add <think> to encourage thinking process before answer


    def _prepare_patient_prompt(self, doctor_question: str, description: str) -> str:
        """
        Prepare a prompt for the environment LLM (patient model) to answer the doctor's question.

        Args:
            doctor_question (str): The doctor's question
            description (str): The patient's self-report states

        Returns:
            str: A formatted prompt for the patient model
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

        # Apply chat template using patient tokenizer
        prompt_text = self.patient_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        return prompt_text


    def run_simulation(self, max_iterations=10, start_idx=0, end_idx=None, batch_size=8):
        """
        运行医患对话模拟，支持批处理。
        Args:
            max_iterations (int): 最大对话轮次.
            start_idx (int): 开始处理的对话索引 (基于原始数据集).
            end_idx (int): 结束处理的对话索引 (不包含) (基于原始数据集).
            batch_size (int): 批处理大小 (主要影响患者模型).
        Returns:
            list: 所有对话的模拟结果列表.
        """
        self.batch_size = batch_size # 更新批处理大小

        dialogue_data = self.load_dialogue_data()

        # 根据 start_idx 和 end_idx 截取需要处理的数据
        if end_idx is None:
            end_idx = len(dialogue_data)
        # 存储原始索引和对应的数据
        dialogue_data_subset = [(i, dialogue_data[i]) for i in range(start_idx, end_idx)]

        # 在分布式环境中，每个进程只处理部分数据
        if global_world_size and global_world_size > 1:
            # 计算每个进程处理的数据量
            per_rank = len(dialogue_data_subset) // global_world_size
            remainder = len(dialogue_data_subset) % global_world_size

            # 计算当前进程的起始和结束索引
            start_idx = global_rank * per_rank + min(global_rank, remainder)
            end_idx = start_idx + per_rank + (1 if global_rank < remainder else 0)

            # 只处理分配给当前进程的数据
            dialogue_data_subset = dialogue_data_subset[start_idx:end_idx]
            print(f"Process {global_rank}: Processing {len(dialogue_data_subset)} items (indices {start_idx}-{end_idx-1})")

        # 初始化活跃对话的状态列表
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

             # 提取原始对话中的问答对，用于患者模型查找回答 (This part seems unused in the current patient model logic)
             # questions_responses = self.extract_doctor_questions_and_patient_responses(dialogue)

             # 初始化对话状态
             active_dialogues.append({
                 "id": original_idx, # 使用原始索引作为对话 ID
                 "self_report": self_report,
                 "true_diagnosis": true_diagnosis,
                 "true_recommendation": true_recommendation,
                 "enhanced_description": enhanced_description,
                 "simulation_dialogue": [], # 存储模拟生成的对话轮次
                 # dialogue_history_messages will store messages in OpenAI chat format [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
                 "dialogue_history_messages": [],
                 "iteration": 0, # 当前对话轮次计数
                 "is_completed": False, # 对话是否已完成
                 # "questions_responses": questions_responses # Store original Q&A pairs (currently unused)
             })

        simulation_results = [] # 用于存储所有已完成对话的最终结果

        print(f"Starting simulation for {len(active_dialogues)} dialogues with batch size {self.batch_size}")

        # Main simulation loop, continue as long as there are active dialogues
        while active_dialogues:
            doctor_turn_dialogues = [] # List of dialogues needing a doctor response
            patient_turn_dialogues = [] # List of dialogues needing a patient response
            finished_dialogues_indices = [] # Indices of completed dialogues in active_dialogues

            # Categorize active dialogues into those needing a doctor or patient response
            for i, state in enumerate(active_dialogues):
                if state['is_completed']:
                    finished_dialogues_indices.append(i)
                    continue # Skip completed dialogues

                # Check if maximum iterations reached
                if state['iteration'] >= max_iterations:
                     # Force final diagnosis
                     final_diagnosis = self.generate_final_diagnosis(state['self_report'], state['dialogue_history_messages'])
                     diagnosis = final_diagnosis.get("diagnosis", "")
                     advice = final_diagnosis.get("advice", "")

                     # Record final diagnosis in simulation dialogue
                     state['simulation_dialogue'].append({
                         "turn": state['iteration'] + 1, # Diagnosis happens after the current iteration
                         "role": "doctor",
                         "content": f"Diagnosis: {diagnosis}\nRecommendation: {advice}",
                         "tokens": final_diagnosis.get("tokens", 0),
                         "prompt_tokens": final_diagnosis.get("prompt_tokens", 0),
                         "is_diagnosis": True,
                         "is_forced": True # Mark as forced completion
                     })
                     state['is_completed'] = True # Mark as completed
                     finished_dialogues_indices.append(i) # Add to completed list
                     continue

                # Determine next turn: doctor or patient
                # If dialogue history is empty, or the last message is from the user (patient), it's the doctor's turn
                if not state['dialogue_history_messages'] or state['dialogue_history_messages'][-1]["role"] == "user":
                    doctor_turn_dialogues.append(state)
                # If the last message is from the assistant (doctor), it's the patient's turn
                elif state['dialogue_history_messages'][-1]["role"] == "assistant":
                     # Need to extract the question the doctor just asked for the patient model
                     last_doctor_message_content = state['dialogue_history_messages'][-1]["content"]
                     processed_last_doctor_message = self.process_doctor_response(last_doctor_message_content)

                     if processed_last_doctor_message["type"] == "question":
                         # Store the doctor's question in the state for patient batch processing
                         state['doctor_question'] = processed_last_doctor_message["content"]
                         patient_turn_dialogues.append(state)
                     else:
                         # If the doctor's last message was not a question (but a diagnosis), the dialogue is completed
                         state['is_completed'] = True
                         finished_dialogues_indices.append(i)


            # Remove completed dialogues from active_dialogues and add to simulation_results
            # Remove from the end to avoid index issues
            for i in sorted(finished_dialogues_indices, reverse=True):
                 completed_dialogue = active_dialogues.pop(i)
                 simulation_results.append({
                     "id": completed_dialogue['id'],
                     "self_report": completed_dialogue['self_report'],
                     "true_diagnosis": completed_dialogue['true_diagnosis'],
                     "true_recommendation": completed_dialogue['true_recommendation'],
                     "simulation_dialogue": completed_dialogue['simulation_dialogue'],
                     "total_turns": completed_dialogue['iteration'],
                     "is_completed": completed_dialogue['is_completed'],
                     "enhanced_description": completed_dialogue.get('enhanced_description', '') # Include enhanced description in results
                 })

            # Process dialogues needing a doctor response (using DeepSeek API - individual calls)
            if doctor_turn_dialogues:
                print(f"Processing {len(doctor_turn_dialogues)} doctor turns...")
                # Process doctor turns one by one as the API call is not inherently batched in this implementation
                for state in tqdm(doctor_turn_dialogues, desc="Doctor Turns", leave=False):
                    # Call the function to generate doctor response for this single state
                    # We pass a list containing only this state to reuse the batch_generate_doctor_responses structure
                    doctor_results = self.batch_generate_doctor_responses([state])

                    # Update the state based on the result (there will be only one result)
                    if doctor_results:
                        result = doctor_results[0]
                        # Find the corresponding state in active_dialogues (needed if the list order changes)
                        current_state = next((d for d in active_dialogues if d['id'] == result["dialogue_id"]), None)
                        if current_state:
                            current_state['iteration'] += 1 # Increment turn count after doctor's response

                            if result["type"] == "question":
                                doctor_question = result["content"]
                                # Record doctor's question in simulation dialogue
                                current_state['simulation_dialogue'].append({
                                    "turn": current_state['iteration'],
                                    "role": "doctor",
                                    "content": doctor_question,
                                    "tokens": result.get("tokens", 0),
                                    "prompt_tokens": result.get("prompt_tokens", 0)
                                })
                                # Add doctor's response to dialogue history for next turn's model input
                                # The history stores messages in OpenAI chat format
                                current_state['dialogue_history_messages'].append({"role": "assistant", "content": result["ori_response"]})

                            else: # If it's a diagnosis
                                diagnosis = result.get("diagnosis", "")
                                advice = result.get("advice", "")
                                # Record diagnosis in simulation dialogue
                                current_state['simulation_dialogue'].append({
                                    "turn": current_state['iteration'],
                                    "role": "doctor",
                                    "content": f"Diagnosis: {diagnosis}\nRecommendation: {advice}",
                                    "tokens": result.get("tokens", 0),
                                    "prompt_tokens": result.get("prompt_tokens", 0),
                                    "is_diagnosis": True
                                })
                                current_state['is_completed'] = True # Dialogue completed


            # Process dialogues needing a patient response (using local model - batched)
            if patient_turn_dialogues:
                 print(f"Processing {len(patient_turn_dialogues)} patient turns in batches...")
                 # Re-collect states needing patient response, as active_dialogues might have changed
                 current_patient_turn_states = []
                 for state in active_dialogues:
                     # Check if dialogue is not completed, history is not empty, and last message is from doctor
                     if not state['is_completed'] and state['dialogue_history_messages'] and state['dialogue_history_messages'][-1]["role"] == "assistant":
                          # Ensure the doctor's last message was a question
                          processed_last_doctor_message = self.process_doctor_response(state['dialogue_history_messages'][-1]["content"])
                          if processed_last_doctor_message["type"] == "question":
                               # Ensure doctor_question is set (should be from the doctor turn processing)
                               if 'doctor_question' in state:
                                   current_patient_turn_states.append(state)
                               else:
                                   print(f"Warning: doctor_question not found for dialogue {state['id']}, skipping patient turn.")


                 # Process in batches
                 for i in tqdm(range(0, len(current_patient_turn_states), self.batch_size), desc="Patient Batches", leave=False):
                     batch_states = current_patient_turn_states[i:i + self.batch_size]
                     # Call the function to generate patient responses for this batch
                     patient_results = self.batch_generate_patient_responses(batch_states)

                     # Update each dialogue's state based on the results
                     for result in patient_results:
                         dialogue_id = result["dialogue_id"]
                         # Find the corresponding active dialogue state
                         state = next((d for d in active_dialogues if d['id'] == dialogue_id), None)
                         if state:
                             patient_response_content = result["content"]
                             # Record patient response in simulation dialogue
                             state['simulation_dialogue'].append({
                                 "turn": state['iteration'], # Patient response corresponds to the doctor's question in this turn
                                 "role": "patient",
                                 "content": patient_response_content
                             })
                             # Add patient response to dialogue history for the next turn's model input
                             # Add instruction for the doctor to continue or diagnose
                             state['dialogue_history_messages'].append({"role": "user", "content": patient_response_content + f"\nTurn {state['iteration'] + 1}/{max_iterations}. You have to give a diagnosis and a recommendation before the end of the conversation."})


            # After processing all batches, check again for newly completed dialogues
            newly_finished_dialogues_indices = []
            for i, state in enumerate(active_dialogues):
                 if state['is_completed']:
                     newly_finished_dialogues_indices.append(i)

            # Remove newly completed dialogues
            for i in sorted(newly_finished_dialogues_indices, reverse=True):
                 completed_dialogue = active_dialogues.pop(i)
                 simulation_results.append({
                     "id": completed_dialogue['id'],
                     "self_report": completed_dialogue['self_report'],
                     "true_diagnosis": completed_dialogue['true_diagnosis'],
                     "true_recommendation": completed_dialogue['true_recommendation'],
                     "simulation_dialogue": completed_dialogue['simulation_dialogue'],
                     "total_turns": completed_dialogue['iteration'],
                     "is_completed": completed_dialogue['is_completed'],
                     "enhanced_description": completed_dialogue.get('enhanced_description', '') # Include enhanced description in results
                 })

            # If no progress was made in this iteration (no doctor or patient turns processed)
            # and there are still active dialogues, break the loop and force final diagnosis
            if not doctor_turn_dialogues and not patient_turn_dialogues and active_dialogues:
                 print("Warning: No progress made in this iteration, breaking loop and forcing final diagnosis for remaining dialogues.")
                 # Force final diagnosis for remaining active dialogues
                 for state in active_dialogues:
                      if not state['is_completed']:
                           final_diagnosis = self.generate_final_diagnosis(state['self_report'], state['dialogue_history_messages'])
                           diagnosis = final_diagnosis.get("diagnosis", "")
                           advice = final_diagnosis.get("advice", "")

                           state['simulation_dialogue'].append({
                               "turn": state['iteration'] + 1,
                               "role": "doctor",
                               "content": f"Diagnosis: {diagnosis}\nRecommendation: {advice}",
                               "tokens": final_diagnosis.get("tokens", 0),
                               "prompt_tokens": final_diagnosis.get("prompt_tokens", 0),
                               "is_diagnosis": True,
                               "is_forced": True
                           })
                           state['is_completed'] = True
                           simulation_results.append({
                               "id": state['id'],
                               "self_report": state['self_report'],
                               "true_diagnosis": state['true_diagnosis'],
                               "true_recommendation": state['true_recommendation'],
                               "simulation_dialogue": state['simulation_dialogue'],
                               "total_turns": state['iteration'],
                               "is_completed": state['is_completed'],
                               "enhanced_description": state.get('enhanced_description', '') # Include enhanced description in results
                           })
                 active_dialogues = [] # Clear the list of active dialogues


        # In a distributed environment, collect results from all processes
        if global_world_size and global_world_size > 1:
            # Gather all results
            all_results = [None] * global_world_size
            dist.all_gather_object(all_results, simulation_results)

            # Only on the main process (rank 0), merge the results
            if global_rank == 0:
                # Flatten the list of results from all processes
                simulation_results = [result for process_results in all_results for result in process_results]
            else:
                # Non-main processes return an empty list
                simulation_results = []

        # Save the final results to a file
        if global_rank is None or global_rank == 0: # Only save from the main process or in non-distributed mode
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(simulation_results, f, ensure_ascii=False, indent=2)

            print(f"Simulation complete. Results saved to: {self.output_file}")

        return simulation_results


def main():
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Medical Dialogue Simulation with DeepSeek-v3 API (Doctor) and Local Model (Patient)")
    parser.add_argument("--doctor_model_name", type=str, default="deepseek-v3-250324",
                        help="DeepSeek-v3 model name for the doctor")
    parser.add_argument("--patient_model_path", type=str, default="Qwen2.5-7B-Instruct",
                        help="Path to the local patient model")
    parser.add_argument("--input_file", type=str, default="ragen/env/medical_consultation/evaluation/test.json",
                        help="Input JSON file containing dialogue data")
    parser.add_argument("--output_dir", type=str,
                        default="outputs/deepseek-v3/", # Modified default output directory
                        help="Output directory for the simulation results")
    parser.add_argument("--output_prefix", type=str, default="test_deepseek_en_output", # Modified output filename prefix
                        help="Prefix for the output filename")
    parser.add_argument("--add_timestamp", action="store_true",
                        help="Add timestamp to output filename")
    parser.add_argument("--max_iterations", type=int, default=10,
                        help="Maximum number of dialogue turns")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for doctor model generation")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling parameter for doctor model")
    # device_map is only used for the local patient model now
    parser.add_argument("--device_map", type=str, default="auto",
                        help="Device map for patient model loading (auto, cuda:0, etc.)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed progress information")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Starting index for dialogue processing")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="Ending index for dialogue processing (None means process to the end)")
    parser.add_argument("--batch_size", type=int, default=8, # Batch size primarily affects patient model
                        help="Batch size for patient model inference")
    # local_rank is handled by the distributed launcher, but kept for compatibility
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training")


    # Add usage example
    parser.epilog = """
    Example usage:
    # Run simulation with DeepSeek-v3 as doctor and Qwen as patient
    # Ensure DEEPSEEK_API_KEY environment variable is set
    python your_script_name.py --doctor_model_name deepseek-v3-250324 --patient_model_path /path/to/qwen_model --input_file input.json --output_dir ./results --add_timestamp --batch_size 16

    # To process only a subset of dialogues:
    python your_script_name.py --start_idx 10 --end_idx 20 --add_timestamp --batch_size 8

    # To run in distributed mode (requires torch.distributed.launch or similar):
    # export DEEPSEEK_API_KEY='your_api_key'
    # python -m torch.distributed.launch --nproc_per_node=2 your_script_name.py ...
    """
    parser.formatter_class = argparse.RawDescriptionHelpFormatter

    args = parser.parse_args()

    if args.verbose:
        print(f"Starting Medical Dialogue Simulation with DeepSeek-v3 API (Doctor) and Local Model (Patient)")
        print(f"Doctor model: {args.doctor_model_name}")
        print(f"Patient model path: {args.patient_model_path}")
        print(f"Input file: {args.input_file}")
        print(f"Temperature: {args.temperature}, Top-p: {args.top_p}")
        print(f"Maximum iterations: {args.max_iterations}")
        print(f"Patient model batch size: {args.batch_size}")
        if args.start_idx > 0 or args.end_idx is not None:
            print(f"Processing dialogues from index {args.start_idx} to {args.end_idx if args.end_idx is not None else 'end'}")

    # Create output filename
    output_filename = args.output_prefix

    # Add index range to filename if specified
    if args.start_idx > 0 or args.end_idx is not None:
        range_info = f"_{args.start_idx}_to_{args.end_idx if args.end_idx is not None else 'end'}"
        output_filename += range_info

    # Add timestamp if needed
    if args.add_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename += f"_{timestamp}"

    output_filename += ".json"

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, output_filename)

    if args.verbose:
        print(f"Output will be saved to: {output_file}")
        print("Initializing models and API client...")

    start_time = time.time()

    # Create and run simulator
    simulator = MedicalDialogueSimulation(
        doctor_model_name=args.doctor_model_name,
        patient_model_path=args.patient_model_path,
        input_file=args.input_file,
        output_file=output_file,
        temperature=args.temperature,
        top_p=args.top_p,
        device_map=args.device_map, # Device map for patient model
        batch_size=args.batch_size # Batch size for patient model
    )

    results = simulator.run_simulation(
        max_iterations=args.max_iterations,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        batch_size=args.batch_size # Pass batch size to run_simulation
    )

    # Calculate and report statistics
    end_time = time.time()
    total_time = end_time - start_time

    # Only report statistics from the main process or in non-distributed mode
    if global_rank is None or global_rank == 0:
        if args.verbose:
            print(f"\nSimulation complete in {total_time:.2f} seconds")
            print(f"Results saved to: {output_file}")

            # Calculate some basic statistics
            if results:
                total_dialogues = len(results)
                # Filter out results that might be None if a process failed or returned empty
                valid_results = [r for r in results if r is not None]
                completed_dialogues = sum(1 for r in valid_results if r.get('is_completed', False))
                # Calculate average turns from valid results
                avg_turns = sum(r.get('total_turns', 0) for r in valid_results) / len(valid_results) if valid_results else 0

                print(f"\nStatistics:")
                print(f"- Total dialogues processed (across all ranks): {total_dialogues}")
                print(f"- Completed dialogues: {completed_dialogues} ({completed_dialogues/total_dialogues*100:.1f}%)")
                print(f"- Average turns per dialogue: {avg_turns:.2f}")
            else:
                 print("\nNo results to report statistics.")


    # In a distributed environment, ensure all processes finish before exiting
    if global_world_size and global_world_size > 1:
        dist.barrier()
        if global_rank == 0:
             print(f"All processes completed.")
        # Clean up distributed environment
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
