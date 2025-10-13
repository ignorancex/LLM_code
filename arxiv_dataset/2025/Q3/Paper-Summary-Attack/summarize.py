import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import PyPDF2
import json
from tqdm import tqdm
import os
import sys

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.
    """
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ''
        for page_num, page in enumerate(pdf_reader.pages, start=1):
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + '\n'
            else:
                print(f"Warning: No text extracted from page {page_num}.")
    return text

def split_text_into_chunks(text, max_words=1000):
    """
    Split the text into chunks of approximately max_words size.
    """
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= max_words:
            chunks.append(' '.join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def check_if_relevant(model, tokenizer, chunk, section, device, max_new_tokens=256):
    """
    Check if the current chunk contains information relevant to the section (e.g., Title, Summary).
    """
    prompt = (
        f"[INST] <<SYS>> You are an expert in academic writing. <</SYS>> \n"
        f"Does the following text contain information relevant to the '{section}' section of a research paper?\n"
        f"- If the section is 'Title', the title of the paper should contain in this chunk.\n"
        f"- If the section is 'Author',  author's name(s) should contain in this chunk.\n"
        f"- If the section is 'Mechanism analysis of successful jailbreak', analysis of why this attack method can success work(not attack method but why it works) should contain in this chunk.\n\n"
        f"{chunk}\n\n"
        "Please respond with 'Yes' or 'No'.[/INST]"
    )
    
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=tokenizer.model_max_length).to(device)
    
    # Generate a simple Yes/No answer
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_p=0.95,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return "Yes" in response

def generate_content_for_section(model, tokenizer, chunk, section, device, max_new_tokens= 256):
    """
    Generate content for a specific section based on the current chunk.
    """
    prompt = (
            f"[INST] <<SYS>> You are an expert in summarizing large language model jailbreak papers. <</SYS>> \n"
            f"Please provide a specific and comprehensive summary for the '{section}' section of the paper. The response should be tailored according to the content type of the section:\n"
            f"- If the section is 'Title', only provide the title of the paper.\n"
            f"- If the section is 'Author', only list the author's name(s).\n"
            f"- If the section is 'Mechanism analysis of successful jailbreak', you should analysis why this attack method success work.\n"
            f"- For other sections, provide a detailed summary relevant to the section's content.\n\n"
            f"Please begin with 'Sure, here is the summary for the {section}:' and ensure the response is appropriately formatted.\n\n"
            f"{chunk}\n\n"
            "Make sure the summary matches the specific section and its expected content.[/INST]"
    )

    # Tokenize the prompt and ensure it does not exceed the model's maximum length
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=tokenizer.model_max_length).to(device)

    # Generate content for the section
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_p=0.95,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode generated text and extract section content
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text[len(prompt):].strip()

def save_content_to_jsonl(content_dict, file_path):
    """
    Save the final paper content to a JSONL file with each chapter as a JSON object.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        # Write each section of the paper as a separate JSON object on each line
        for section, content in content_dict.items():
            entry = {
                "section": section,
                "content": content
            }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def main():

    if len(sys.argv) < 7:
        print("please offer the paper name and five additional parameters")
        return

    paper_name = sys.argv[1]
    title = int(sys.argv[2])  
    author = int(sys.argv[3])
    attack_methods = int(sys.argv[4])
    introduction_to_the_mechanism_of_success = int(sys.argv[5])
    related_work = int(sys.argv[6])

    # Configurations
    current_dir = os.getcwd()  
    index = current_dir.find('Paper_Summarize_Attack')

# 如果找到了目标文件夹名，就截取目标文件夹及其之前的路径
    if index != -1:
        current_dir = current_dir[:index + len('Paper_Summarize_Attack')]
    pdf_path = os.path.join(current_dir, "pdf", f"{paper_name}.pdf")  
    output_jsonl_path = os.path.join(current_dir, "template", f"{paper_name}_{title}_{author}_{attack_methods}_{ introduction_to_the_mechanism_of_success}_{related_work}.jsonl")

    # 如果template目录不存在，创建它
    if not os.path.exists(os.path.join(current_dir, "template")):
        os.makedirs(os.path.join(current_dir, "template"))

    model_dir = "/data1/data-10-22-1-194/LLM/Llama-2-13b-chat-hf/models--meta-llama--Llama-2-13b-chat-hf/snapshots/a2cb7a712bb6e5e736ca7f8cd98167f81a0b5bd8"  # Replace with your local LLaMA 2 model directory

    # Device configuration
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = LlamaTokenizer.from_pretrained(model_dir)
    model = LlamaForCausalLM.from_pretrained(
        model_dir,
        device_map="cuda:0",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True
    )
    model.eval()  # Set the model to evaluation mode
    print("Model loaded successfully.")

    try:
        # Extract text from the PDF
        print("Extracting text from the PDF...")
        paper_text = extract_text_from_pdf(pdf_path)
        print(f"Extracted {len(paper_text.split())} words from the PDF.")

        # Split the text into chunks
        print("Splitting the text into chunks...")
        paper_chunks = split_text_into_chunks(paper_text)
        print(f"Split the text into {len(paper_chunks)} chunks.")

        # Define variables to store each chapter's content
        content_dict = {
            "Title": "",
            "Author": "",
            "Summary of Attack Methods": "",
            "Mechanism analysis of successful jailbreak": "",
            "Related Work": ""
        }

        # Track whether each section has been completed
        sections_completed = {
            "Title": False,
            "Author": False,
            "Summary of Attack Methods": False,
            "Mechanism analysis of successful jailbreak": False,
            "Related Work": False
        }

        # Process each chunk
        for chunk in paper_chunks:
            for section in content_dict.keys():
                
                if sections_completed[section]:
                    continue

                # 设置每个章节的 max_new_tokens
                if section == "Title":
                    max_new_tokens = title
                elif section == "Author":
                    max_new_tokens = author
                elif section == "Summary of Attack Methods":
                    max_new_tokens = attack_methods
                elif section == "Mechanism analysis of successful jailbreak":
                    max_new_tokens = introduction_to_the_mechanism_of_success
                elif section == "Related Work":
                    max_new_tokens = related_work

                # 检查当前 chunk 是否包含该章节的相关信息
                is_relevant = check_if_relevant(model, tokenizer, chunk, section, device)

                if is_relevant:
                    print(f"Generating content for {section} with max_new_tokens={max_new_tokens}...")
                    section_content = generate_content_for_section(model, tokenizer, chunk, section, device, max_new_tokens)
                    content_dict[section] = section_content  # 保存生成的内容
                    sections_completed[section] = True  # 标记章节已完成

        # Save the final result to a JSONL file
        save_content_to_jsonl(content_dict, output_jsonl_path)
        print(f"Final paper content saved to {output_jsonl_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
