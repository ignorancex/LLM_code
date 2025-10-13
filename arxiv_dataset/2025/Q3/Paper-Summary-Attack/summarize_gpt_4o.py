import openai
import PyPDF2
import json
import os
import sys

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
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

# Function to split text into chunks
def split_text_into_chunks(text, max_words=1000):
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

# Function to call GPT-4 model via OpenAI API to check relevance
def check_if_relevant(chunk, section, api_key, base_url, model="gpt-4o"):
    prompt = (
        f"You are an expert in academic writing. Does the following text contain information relevant to the '{section}' section of a research paper?\n\n"
        f"- If the section is 'Mechanism analysis of successful jailbreak', analysis of why this attack method can success work(not attack method but why it works) should contain in this chunk.\n\n"
        f"{chunk}\n\n"
        "Please respond with 'Yes' or 'No'."
    )

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "system", "content": prompt}],
            api_key=api_key,
            base_url=base_url,
            temperature=0.3,
        )
        result = response['choices'][0]['message']['content'].strip()
        return "Yes" in result
    except Exception as e:
        print(f"An error occurred while checking relevance: {e}")
        return False

# Function to call GPT-4 model via OpenAI API to generate content
def generate_content_for_section(chunk, section, api_key, base_url, model="gpt-4o", max_tokens= 256):
    prompt = (
            f"You are an expert in summarizing large language model jailbreak papers.\n"
            f"Please provide a specific and comprehensive summary for the '{section}' section of the paper. The response should be tailored according to the content type of the section:\n"
            f"- If the section is 'Title', only provide the title of the paper.\n"
            f"- If the section is 'Author', only list the author's name(s).\n"
            f"- If the section is 'Mechanism analysis of successful jailbreak', you should analysis why this attack method success work.\n"
            f"- For other sections, provide a detailed summary relevant to the section's content.\n\n"
            f"Please begin with 'Sure, here is the summary for the {section}:' and ensure the response is appropriately formatted.\n\n"
            f"{chunk}\n\n"
            "Make sure the summary matches the specific section and its expected content."
    )

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "system", "content": prompt}],
            api_key=api_key,
            base_url=base_url,
            temperature=0.3,
            max_tokens=max_tokens
        )
        result = response['choices'][0]['message']['content'].strip()
        return result
    except Exception as e:
        print(f"An error occurred while generating content: {e}")
        return ""

# Function to save content to JSONL file
def save_content_to_jsonl(content_dict, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for section, content in content_dict.items():
            entry = {"section": section, "content": content}
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

# Main function
def main():
    if len(sys.argv) < 7:
        print("Please offer the paper name and five additional parameters.")
        return

    paper_name = sys.argv[1]
    title_tokens = int(sys.argv[2])
    author_tokens = int(sys.argv[3])
    attack_methods_tokens = int(sys.argv[4])
    intro_mechanism_tokens = int(sys.argv[5])
    related_work_tokens = int(sys.argv[6])

    api_key = "aaa"  # Replace with your actual OpenAI API key
    base_url = "http://47.74.22.128:35040/openai/v1/"  # Replace with your actual base URL

    # Load the PDF
    current_dir = os.getcwd()  
    index = current_dir.find('Paper_Summarize_Attack')
    if index != -1:
        current_dir = current_dir[:index + len('Paper_Summarize_Attack')]
    pdf_path = os.path.join(current_dir, "pdf", f"{paper_name}.pdf")
    output_jsonl_path = os.path.join(current_dir, "template", f"{paper_name}_output.jsonl")

    # Ensure template directory exists
    if not os.path.exists(os.path.join(current_dir, "template")):
        os.makedirs(os.path.join(current_dir, "template"))

    try:
        # Extract text from PDF
        print("Extracting text from the PDF...")
        paper_text = extract_text_from_pdf(pdf_path)
        print(f"Extracted {len(paper_text.split())} words from the PDF.")

        # Split the text into chunks
        print("Splitting the text into chunks...")
        paper_chunks = split_text_into_chunks(paper_text)
        print(f"Split the text into {len(paper_chunks)} chunks.")

        # Dictionary to store the content of each section
        content_dict = {
            "Title": "",
            "Author": "",
            "Summary of Attack Methods": "",
            "Mechanism analysis of successful jailbreak": "",
            "Related Work": ""
        }

        # Process each chunk
        for chunk in paper_chunks:
            for section in content_dict.keys():
                # Set the token limit for each section
                if section == "Title":
                    max_tokens = title_tokens
                elif section == "Author":
                    max_tokens = author_tokens
                elif section == "Summary of Attack Methods":
                    max_tokens = attack_methods_tokens
                elif section == "Mechanism analysis of successful jailbreak":
                    max_tokens = intro_mechanism_tokens
                elif section == "Related Work":
                    max_tokens = related_work_tokens

                # Check if the chunk is relevant to the current section
                is_relevant = check_if_relevant(chunk, section, api_key, base_url)

                if is_relevant:
                    print(f"Generating content for {section}...")
                    section_content = generate_content_for_section(chunk, section, api_key, base_url, max_tokens=max_tokens)
                    content_dict[section] = section_content
                else :
                    print("NO")
        # Save the final result to a JSONL file
        save_content_to_jsonl(content_dict, output_jsonl_path)
        print(f"Final paper content saved to {output_jsonl_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
