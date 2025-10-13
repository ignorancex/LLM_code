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

class MedicalDialogueSimulation:
    def __init__(self, model_path, input_file, output_file, temperature=0.7, top_p=0.9, device_map="auto", batch_size=8):
        """
        初始化医患对话模拟系统
        Args:
            model_path (str): 医生模型的路径.
            input_file (str): 包含对话数据的输入 JSON 文件路径.
            output_file (str): 保存模拟结果的输出 JSON 文件路径.
            temperature (float): 医生模型生成时的温度参数.
            top_p (float): 医生模型生成时的 top-p 参数.
            device_map (str): 模型加载的设备映射.
            batch_size (int): 批处理大小.
        """
        self.input_file = input_file
        self.output_file = output_file
        self.temperature = temperature
        self.top_p = top_p
        self.batch_size = batch_size # 存储批处理大小

        # 设置分布式环境
        is_distributed = setup_distributed()
        if is_distributed:
            self.device = torch.device(f"cuda:{global_local_rank}")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Process {global_rank if is_distributed else 0}: Using device: {self.device}")

        # 加载医生模型和分词器
        # 设置 padding_side='left' for decoder-only models
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
        # 确保 padding token 已设置，用于批处理
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # 在分布式环境中，每个进程加载模型到自己的GPU
        if is_distributed:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=None,  # 不使用device_map，手动管理
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            self.model.to(self.device)
        else:
            # 非分布式环境，使用device_map自动管理
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device_map,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )

        # 加载患者模型和分词器 (假设患者模型使用 Qwen2.5-7B-Instruct)
        # 注意：如果患者模型路径不同，请修改这里
        patient_model_path = "Qwen2.5-7B-Instruct"
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

        # 医生系统提示词
        self.doctor_system_prompt = """You are an experienced doctor who needs to provide professional diagnosis and advice to patients through consultation. Please listen carefully to the patient's description, ask targeted questions, and collect sufficient information before giving a diagnosis and treatment recommendation.

Quick Guide
Objectives:
1. Obtain key information through effective questioning, each round of questions should be modified based on the previous round's content, meaning you shouldn't ask similar questions.
2. Comprehensively analyze the patient's condition to provide an accurate diagnosis and appropriate treatment recommendations.

Rules:
1. You can only choose one of the options to respond, you cannot both answer questions and provide a diagnosis simultaneously.
2. Absolutely do not repeat or ask questions similar or identical to those previously asked.

Response:
<think> [your thinking] </think>
<answer>If you believe there is insufficient information, **please only ask one question**, in this format:
Question: (your question).
</answer> | <answer>If you believe you have obtained enough information, please only provide diagnosis and recommendations, in this format:
Diagnosis: (the patient's most likely disease or symptoms)
Recommendation: (corresponding treatment plan or advice)
</answer>
Rewards:
Incorrect format: -2.0
Effective question (patient can provide an answer and the question is helpful for diagnosis): +1.0
Ineffective questions do not count towards score
Repeated questions: -2.0
The number of conversation turn is limited. Reaching maximum interaction rounds without providing a diagnosis: -5.0
Completely correct diagnosis and recommendations: +10.0
      """

        # 最终诊断提示词
        self.final_diagnosis_system_prompt = """You are an experienced doctor who must provide a diagnosis and recommendation based on existing information. You have already asked enough questions and must now give the final diagnosis and treatment advice.

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
            advice_match = re.search(r"Recommendation[:：](.*?)(?=\n|$)", doctor_response, re.DOTALL)

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
        """计算文本的token数量"""
        # 使用医生模型的分词器计算 token 数量
        return len(self.tokenizer.encode(text))

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

        # 应用聊天模板并分词
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
                max_new_tokens=512,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id # 确保 pad token 被正确处理
            )

        # 解码响应并提取诊断和建议
        response_tokens = outputs.sequences[0].size(0) - inputs.input_ids.size(1)
        response = self.tokenizer.decode(outputs.sequences[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

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

        # 为批处理准备医生模型的输入 prompt
        batched_prompts = []
        for state in dialogue_states:
            # 构建当前轮次的 prompt
            # cur = f"""Current question: {state['self_report']}
            cur = f"""\nPatient's description: {state['self_report']}
            Decide next action:
            Always output: <think> [your thinking] </think> <answer> [your response] </answer> No additional text. Strictly follow this format.
            """
            messages = [
                # {"role": "system", "content": self.doctor_system_prompt},
                {"role": "user", "content": self.doctor_system_prompt + cur}
            ]
            # 添加对话历史
            messages.extend(state['dialogue_history_messages'])
            # 应用聊天模板
            formatted_input = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            # 添加 <think> 以引导模型思考过程
            formatted_input = formatted_input + "<think>"
            batched_prompts.append(formatted_input)

        # 对 prompt 进行分词和 padding，以便进行批处理
        # padding='longest' 会将所有序列 padding 到当前 batch 中最长序列的长度
        # truncation=True 会截断过长的序列
        inputs = self.tokenizer(batched_prompts, return_tensors="pt", padding='longest', truncation=True).to(self.model.device)

        # 计算每个 prompt 的 token 数量 (在 padding 之前)
        doctor_prompt_tokens = [self.count_tokens(p) for p in batched_prompts]

        # 批量生成医生回复
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512, # 控制生成的最大 token 数量
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id # 确保 pad token 被正确处理
            )

        # 解码批量生成的响应
        # 切片以移除输入的 prompt token
        batched_sequences = outputs.sequences
        decoded_responses = self.tokenizer.batch_decode(batched_sequences[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # 处理并返回每个对话的结果
        processed_results = []
        for i, response in enumerate(decoded_responses):
            processed_response = self.process_doctor_response(response.strip())
            # 添加 token 统计信息
            processed_response["prompt_tokens"] = doctor_prompt_tokens[i]
            processed_response["tokens"] = batched_sequences[i].size(0) - inputs.input_ids.shape[1] # 生成的 token 数量

            # 包含原始对话的状态信息，以便后续更新
            processed_response["dialogue_id"] = dialogue_states[i]['id']
            processed_response["ori_response"] = "<think>" + response

            processed_results.append(processed_response)

        return processed_results

    def batch_generate_patient_responses(self, dialogue_states):
        """
        批量生成患者的回复。
        Args:
            dialogue_states (list): 需要生成患者回复的对话状态列表。
                                    每个状态包含: 'id', 'doctor_question', 'questions_responses' (原始数据集的问答对), 'conversation_history' (当前对话历史),
                                    以及可选的 'enhanced_description' 和 'dialogue_history_messages'.
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
            conversation_history = state.get('dialogue_history_messages', [])[:-1] # 不包括当前医生问题
            description = state.get('enhanced_description', "")

            judge_prompt = self._prepare_judge_prompt(doctor_question, conversation_history)
            judge_prompts.append(judge_prompt)
            patient_prompts.append(self._prepare_patient_prompt(doctor_question, description))
            state_indices.append(index)
            dialogue_ids.append(dialogue_id)
            descriptions.append(description)
            conversation_histories.append(conversation_history)

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
                "original_state_index": state_indices[i]
            })

        return processed_results

    def _batch_generate(self, prompts: List[str], max_new_tokens: int) -> List[str]:
        """
        批量生成文本。

        Args:
            prompts (List[str]): 输入的提示列表。
            max_new_tokens (int): 生成的最大 token 数。

        Returns:
            List[str]: 生成的文本列表。
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
                history_questions.append(conv['extract_response'])
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
        运行医患对话模拟，支持批处理。
        Args:
            max_iterations (int): 最大对话轮次.
            start_idx (int): 开始处理的对话索引 (基于原始数据集).
            end_idx (int): 结束处理的对话索引 (不包含) (基于原始数据集).
            batch_size (int): 批处理大小.
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

             # 提取原始对话中的问答对，用于患者模型查找回答
             questions_responses = self.extract_doctor_questions_and_patient_responses(dialogue)

             # 初始化对话状态
             active_dialogues.append({
                 "id": original_idx, # 使用原始索引作为对话 ID
                 "self_report": self_report,
                 "true_diagnosis": true_diagnosis,
                 "true_recommendation": true_recommendation,
                 "enhanced_description": enhanced_description,
                 "simulation_dialogue": [], # 存储模拟生成的对话轮次
                 "dialogue_history_messages": [], # 存储用于模型输入的对话历史 (格式为 [{"role": ..., "content": ...}])
                 "iteration": 0, # 当前对话轮次计数
                 "is_completed": False, # 对话是否已完成
                 "questions_responses": questions_responses # 存储原始问答对
             })

        simulation_results = [] # 用于存储所有已完成对话的最终结果

        print(f"Starting simulation for {len(active_dialogues)} dialogues with batch size {self.batch_size}")

        # 主模拟循环，只要还有活跃对话就继续
        while active_dialogues:
            doctor_turn_dialogues = [] # 需要医生进行下一轮回复的对话列表
            patient_turn_dialogues = [] # 需要患者进行下一轮回复的对话列表
            finished_dialogues_indices = [] # 已完成对话在 active_dialogues 中的索引

            # 遍历活跃对话，将它们分类到需要医生回复或患者回复的列表中
            for i, state in enumerate(active_dialogues):
                if state['is_completed']:
                    finished_dialogues_indices.append(i)
                    continue # 跳过已完成的对话

                # 检查是否达到最大轮次
                if state['iteration'] >= max_iterations:
                     # 强制进行最终诊断
                     final_diagnosis = self.generate_final_diagnosis(state['self_report'], state['dialogue_history_messages'])
                     diagnosis = final_diagnosis.get("diagnosis", "")
                     advice = final_diagnosis.get("advice", "")

                     # 记录最终诊断到模拟对话中
                     state['simulation_dialogue'].append({
                         "turn": state['iteration'] + 1, # 诊断发生在当前轮次之后
                         "role": "doctor",
                         "content": f"Diagnosis: {diagnosis}\nRecommendation: {advice}",
                         "tokens": final_diagnosis.get("tokens", 0),
                         "prompt_tokens": final_diagnosis.get("prompt_tokens", 0),
                         "is_diagnosis": True,
                         "is_forced": True # 标记为强制完成
                     })
                     state['is_completed'] = True # 标记为已完成
                     finished_dialogues_indices.append(i) # 添加到已完成列表
                     continue

                # 判断下一轮是医生还是患者
                # 如果对话历史为空，或者最后一条消息是患者的回复，则下一轮是医生回复
                if not state['dialogue_history_messages'] or state['dialogue_history_messages'][-1]["role"] == "user":
                    doctor_turn_dialogues.append(state)
                # 如果最后一条消息是医生的回复，则下一轮是患者回复
                elif state['dialogue_history_messages'][-1]["role"] == "assistant":
                     # 需要提取医生刚刚提出的问题，以便患者模型使用
                     last_doctor_message_content = state['dialogue_history_messages'][-1]["content"]
                     processed_last_doctor_message = self.process_doctor_response(last_doctor_message_content)

                     if processed_last_doctor_message["type"] == "question":
                         # 将医生问题存储在状态中，供患者批处理使用
                         state['doctor_question'] = processed_last_doctor_message["content"]
                         patient_turn_dialogues.append(state)
                     else:
                         # 如果医生最后一条消息不是问题 (而是诊断)，则对话已完成
                         state['is_completed'] = True
                         finished_dialogues_indices.append(i)


            # 将已完成的对话从 active_dialogues 中移除，并添加到 simulation_results
            # 从后往前移除，避免索引问题
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
                     "enhanced_description": completed_dialogue['enhanced_description']
                 })

            # 批量处理需要医生回复的对话
            if doctor_turn_dialogues:
                print(f"Processing {len(doctor_turn_dialogues)} doctor turns in batches...")
                # 按照 batch_size 进行分批处理
                for i in tqdm(range(0, len(doctor_turn_dialogues), self.batch_size), desc="Doctor Batches", leave=False):
                    batch_states = doctor_turn_dialogues[i:i + self.batch_size]
                    # 调用批量生成医生回复的函数
                    doctor_results = self.batch_generate_doctor_responses(batch_states)

                    # 根据批量结果更新每个对话的状态
                    for result in doctor_results:
                        dialogue_id = result["dialogue_id"]
                        # 根据 ID 找到对应的活跃对话状态
                        state = next((d for d in active_dialogues if d['id'] == dialogue_id), None)
                        if state:
                            state['iteration'] += 1 # 医生回复后，轮次加一

                            if result["type"] == "question":
                                doctor_question = result["content"]
                                # 记录医生问题到模拟对话中
                                state['simulation_dialogue'].append({
                                    "turn": state['iteration'],
                                    "role": "doctor",
                                    "content": doctor_question,
                                    "tokens": result.get("tokens", 0),
                                    "prompt_tokens": result.get("prompt_tokens", 0)
                                })
                                # 将医生问题添加到对话历史，用于下一轮模型输入
                                state['dialogue_history_messages'].append({"role": "assistant", "content": result["ori_response"], "extract_response": doctor_question})

                            else: # 如果是诊断
                                diagnosis = result.get("diagnosis", "")
                                advice = result.get("advice", "")
                                # 记录诊断到模拟对话中
                                state['simulation_dialogue'].append({
                                    "turn": state['iteration'],
                                    "role": "doctor",
                                    "content": f"Diagnosis: {diagnosis}\nRecommendation: {advice}",
                                    "tokens": result.get("tokens", 0),
                                    "prompt_tokens": result.get("prompt_tokens", 0),
                                    "is_diagnosis": True
                                })
                                state['is_completed'] = True # 对话完成


            # 批量处理需要患者回复的对话
            if patient_turn_dialogues:
                 print(f"Processing {len(patient_turn_dialogues)} patient turns in batches...")
                 # 重新收集需要患者回复的对话状态，因为 active_dialogues 列表可能已改变
                 current_patient_turn_states = []
                 for state in active_dialogues:
                     # 检查对话是否未完成，且对话历史不为空，且最后一条是医生消息
                     if not state['is_completed'] and state['dialogue_history_messages'] and state['dialogue_history_messages'][-1]["role"] == "assistant":
                          # 确保医生最后一条消息是问题
                          processed_last_doctor_message = self.process_doctor_response(state['dialogue_history_messages'][-1]["content"])
                          if processed_last_doctor_message["type"] == "question":
                               # 确保 doctor_question 已设置 (在上面的医生批处理阶段设置)
                               current_patient_turn_states.append(state)

                 # 按照 batch_size 进行分批处理
                 for i in tqdm(range(0, len(current_patient_turn_states), self.batch_size), desc="Patient Batches", leave=False):
                     batch_states = current_patient_turn_states[i:i + self.batch_size]
                     # 调用批量生成患者回复的函数
                     patient_results = self.batch_generate_patient_responses(batch_states)

                     # 根据批量结果更新每个对话的状态
                     for result in patient_results:
                         dialogue_id = result["dialogue_id"]
                         # 根据 ID 找到对应的活跃对话状态
                         state = next((d for d in active_dialogues if d['id'] == dialogue_id), None)
                         if state:
                             patient_response_content = result["content"]
                             # 记录患者回复到模拟对话中
                             state['simulation_dialogue'].append({
                                 "turn": state['iteration'], # 患者回复对应医生提问的轮次
                                 "role": "patient",
                                 "content": patient_response_content
                             })
                             # 将患者回复添加到对话历史，用于下一轮模型输入
                             # 同时添加引导医生继续或诊断的指令
                             state['dialogue_history_messages'].append({"role": "user", "content": patient_response_content + f"\nTurn {state['iteration'] + 1}/{max_iterations}. You have to give a diagnosis and a recommendation before the end of the conversation."})


            # 在处理完所有批次后，再次检查是否有对话已完成
            newly_finished_dialogues_indices = []
            for i, state in enumerate(active_dialogues):
                 if state['is_completed']:
                     newly_finished_dialogues_indices.append(i)

            # 移除新完成的对话
            for i in sorted(newly_finished_dialogues_indices, reverse=True):
                 completed_dialogue = active_dialogues.pop(i)
                 simulation_results.append({
                     "id": completed_dialogue['id'],
                     "self_report": completed_dialogue['self_report'],
                     "true_diagnosis": completed_dialogue['true_diagnosis'],
                     "true_recommendation": completed_dialogue['true_recommendation'],
                     "simulation_dialogue": completed_dialogue['simulation_dialogue'],
                     "total_turns": completed_dialogue['iteration'],
                     "is_completed": completed_dialogue['is_completed']
                 })

            # 如果在本轮迭代中没有处理任何对话 (例如，所有对话都在批处理中间完成了)，则跳出循环
            # 这也可以防止在出现意外情况时陷入死循环
            if not doctor_turn_dialogues and not patient_turn_dialogues and active_dialogues:
                 print("Warning: No progress made in this iteration, breaking loop and forcing final diagnosis for remaining dialogues.")
                 # 强制对剩余的活跃对话进行最终诊断
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
                               "is_completed": state['is_completed']
                           })
                 active_dialogues = [] # 清空活跃对话列表


        # 在分布式环境中，收集所有进程的结果
        if global_world_size and global_world_size > 1:
            # 收集所有进程的结果
            all_results = [None] * global_world_size
            dist.all_gather_object(all_results, simulation_results)
            
            # 只在主进程(rank 0)上合并结果
            if global_rank == 0:
                # 展平所有进程的结果
                simulation_results = [result for process_results in all_results for result in process_results]
            else:
                # 非主进程返回空列表
                simulation_results = []

        # 将所有对话的最终结果保存到文件
        if global_rank is None or global_rank == 0:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(simulation_results, f, ensure_ascii=False, indent=2)

            print(f"Simulation complete. Results saved to: {self.output_file}")

        return simulation_results


def main():
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="Medical Dialogue Simulation with Qwen 2.5 (Batched)")
    parser.add_argument("--model_path", type=str, default="Qwen2.5-7B-Instruct",
                        help="Path to the Qwen model")
    parser.add_argument("--input_file", type=str, default="ragen/env/medical_consultation/evaluation/test.json",
                        help="Input JSON file containing dialogue data")
    parser.add_argument("--output_dir", type=str,
                        default="outputs/Qwen2.5-7B-Instruct/",
                        help="Output directory for the simulation results")
    parser.add_argument("--output_prefix", type=str, default="test_qwen_en_output_batched", # 修改输出文件名前缀以区分
                        help="Prefix for the output filename")
    parser.add_argument("--add_timestamp", action="store_true",
                        help="Add timestamp to output filename")
    parser.add_argument("--max_iterations", type=int, default=10,
                        help="Maximum number of dialogue turns")
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
    parser.add_argument("--batch_size", type=int, default=8, # 添加批处理大小参数
                        help="Batch size for model inference")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training")


    # 添加使用示例
    parser.epilog = """
    Example usage:
    python your_script_name.py --model_path /path/to/model --input_file input.json --output_dir ./results --add_timestamp --batch_size 16

    To process only a subset of dialogues:
    python your_script_name.py --start_idx 10 --end_idx 20 --add_timestamp --batch_size 8
    """
    parser.formatter_class = argparse.RawDescriptionHelpFormatter

    args = parser.parse_args()

    if args.verbose:
        print(f"Starting Medical Dialogue Simulation with Qwen 2.5 (Batched)")
        print(f"Model path: {args.model_path}")
        print(f"Input file: {args.input_file}")
        print(f"Temperature: {args.temperature}, Top-p: {args.top_p}")
        print(f"Maximum iterations: {args.max_iterations}")
        print(f"Batch size: {args.batch_size}")
        if args.start_idx > 0 or args.end_idx is not None:
            print(f"Processing dialogues from index {args.start_idx} to {args.end_idx if args.end_idx is not None else 'end'}")

    # 创建输出文件名
    output_filename = args.output_prefix

    # 如果指定了索引范围，添加到文件名中
    if args.start_idx > 0 or args.end_idx is not None:
        range_info = f"_{args.start_idx}_to_{args.end_idx if args.end_idx is not None else 'end'}"
        output_filename += range_info

    # 如果需要，添加时间戳
    if args.add_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename += f"_{timestamp}"

    output_filename += ".json"

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, output_filename)

    if args.verbose:
        print(f"Output will be saved to: {output_file}")
        print("Loading model and tokenizer...")

    start_time = time.time()

    # 创建并运行模拟器
    simulator = MedicalDialogueSimulation(
        model_path=args.model_path,
        input_file=args.input_file,
        output_file=output_file,
        temperature=args.temperature,
        top_p=args.top_p,
        device_map=args.device_map,
        batch_size=args.batch_size # 传递批处理大小
    )

    results = simulator.run_simulation(
        max_iterations=args.max_iterations,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        batch_size=args.batch_size # 传递批处理大小
    )

    # 计算并报告统计信息
    end_time = time.time()
    total_time = end_time - start_time

    if args.verbose:
        print(f"\nSimulation complete in {total_time:.2f} seconds")
        print(f"Results saved to: {output_file}")

        # 计算一些基本统计信息
        if results:
            total_dialogues = len(results)
            completed_dialogues = sum(1 for r in results if r.get('is_completed', False))
            # 从结果列表中计算平均轮次
            avg_turns = sum(r.get('total_turns', 0) for r in results) / total_dialogues if total_dialogues > 0 else 0

            print(f"\nStatistics:")
            print(f"- Total dialogues processed: {total_dialogues}")
            print(f"- Completed dialogues: {completed_dialogues} ({completed_dialogues/total_dialogues*100:.1f}%)")
            print(f"- Average turns per dialogue: {avg_turns:.2f}")

    # 在分布式环境中，确保所有进程都完成
    if global_world_size and global_world_size > 1:
        dist.barrier()
        print(f"Process {global_rank}: Simulation completed")


if __name__ == "__main__":
    main()
