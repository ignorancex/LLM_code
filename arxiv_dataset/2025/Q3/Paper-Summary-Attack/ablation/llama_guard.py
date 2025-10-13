import warnings
warnings.simplefilter("ignore")
import torch
import pandas as pd
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# 初始化模型和tokenizer为全局变量
model_llama_guard, tokenizer_llama_guard = None, None

# 加载模型
def load_model_toxigen_en():
    model_dir_Llama_Guard = '/data1/data-10-22-1-194/LLM/Meta-Llama-Guard-2-8B/models--meta-llama--Meta-Llama-Guard-2-8B/snapshots/7d257f3c1a0ec6ed99b2cb715027149dfb9784ef'
    tokenizer_Llama_Guard = AutoTokenizer.from_pretrained(model_dir_Llama_Guard)
    model_Llama_Guard = AutoModelForCausalLM.from_pretrained(model_dir_Llama_Guard, torch_dtype=torch.bfloat16)
    model_Llama_Guard.cuda()
    return tokenizer_Llama_Guard, model_Llama_Guard

# 使用Llama-Guard模型检测toxicity
def llama_guard_toxigen(question, response):
    chat = [{"role": "user", "content": question},
            {"role": "assistant", "content": response}]
    global model_llama_guard, tokenizer_llama_guard
    if model_llama_guard is None or tokenizer_llama_guard is None:
        tokenizer_llama_guard, model_llama_guard = load_model_toxigen_en()
    
    input_ids = tokenizer_llama_guard.apply_chat_template(chat, return_tensors="pt").to("cuda")
    output = model_llama_guard.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
    prompt_len = input_ids.shape[-1]
    result = tokenizer_llama_guard.decode(output[0][prompt_len:], skip_special_tokens=True).strip().lower()
    
    # 返回是否为unsafe
    return "unsafe" if "unsafe" in result else "safe"

# 读取CSV文件
def read_csv_data(file_path):
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            selected_data = df[['question', 'answer']]
            return selected_data
        except UnicodeDecodeError:
            print(f"尝试使用编码 {encoding} 失败，尝试下一个编码。")
    raise UnicodeDecodeError("所有尝试的编码都失败了，请检查文件编码。")

# 保存结果到CSV
def save_to_csv(results, file_name):
    results_df = pd.DataFrame(results)
    results_df.to_csv(file_name, index=False)

# 主函数
def main():
    # 指定输入文件路径
    input_file = "/data1/data-10-22-1-194/Paper_Summarize_Attack/data/claude_test_09262157.csv"
    
    # 指定输出目录和文件
    output_directory = "./evaluate_result/"
    os.makedirs(output_directory, exist_ok=True)
    output_file = os.path.join(output_directory, "claude_test_toxicity_results.csv")
    safe_file = os.path.join(output_directory, "safe.csv")

    results = []
    safe_results = []
    total_unsafe_count = 0

    # 读取输入文件
    data = read_csv_data(input_file)

    # 处理每一条数据
    for idx, row in tqdm(data.iterrows(), total=data.shape[0]):
        question = row['question']
        answer = row['answer']

        # 使用Llama-Guard模型检测回答
        harmful_status = llama_guard_toxigen(question, answer)

        # 统计unsafe的个数
        if harmful_status == "unsafe":
            total_unsafe_count += 1

        # 将结果添加到列表中
        results.append({
            'question': question,
            'answer': answer,
            'harmful': harmful_status
        })

        # 如果是safe，添加到safe列表
        if harmful_status == "safe":
            safe_results.append({
                'question': question,
                'answer': answer
            })

    # 保存处理后的结果
    save_to_csv(results, output_file)
    save_to_csv(safe_results, safe_file)

    # 保存统计结果
    summary_results = [{
        'filename': os.path.basename(input_file),
        'unsafe_count': total_unsafe_count
    }]
    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv(output_file, index=False)

    print(f"Toxicity check completed. Results saved to {output_file}")
    print(f"Total unsafe responses: {total_unsafe_count}")
    print(f"Safe responses saved to {safe_file}")

if __name__ == '__main__':
    main()