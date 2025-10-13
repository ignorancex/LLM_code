from typing import List, Dict
import vllm
from collections import defaultdict
from config import parse_args
import time
import torch
from vllm import SamplingParams
import re


args = parse_args()

sampling_params = SamplingParams(temperature=0.90, top_p=0.9, max_tokens = 4000, seed=args.seed, stop = '*** END')

LLM = vllm.LLM(model=args.model, tensor_parallel_size=args.gpu, gpu_memory_utilization=0.9,trust_remote_code=True)


prompt_summary_analysis = """
### Instructions:

You are given a question and several pieces of evidence. Your task is to analyze the evidence and provide a concise answer to the question. 

For each piece of evidence, the background of its source media is provided. When evaluating the evidence, it is crucial to take into account the credibility of the source media, as this can significantly influence the reliability of the evidence. Additionally, consider any potential biases that may be inherent in the source media, especially if they are explicitly mentioned. This will help ensure a more nuanced and thorough evaluation of the evidence, factoring in both the content and the context in which it is presented.

"""

def write_to_file(file_path, data, append=True):
    mode = 'a' if append else 'w'
    with open(file_path, mode, encoding='utf-8') as file:
        file.write(data + "\n")

def analysis_preprocess(retrieved_results):
    qns = []
    sentences = []
    media_backgrounds = []
    for entity in retrieved_results:
        qns.append(entity['question'])

        contexts = []
        media_bg = []
        for context in entity['top_k']:
            contexts.append(context['sentence'])
            media_bg.append(context.get("details", "None"))

        sentences.append(contexts)
        media_backgrounds.append(media_bg)

    return qns, sentences, media_backgrounds


def extract_before_final_answer_or_end(text):
    # 先尝试匹配 "### Credibility:" 和 "final answer" 之间的内容，兼容大小写
    pattern_credibility_to_final_answer = r"###\s*Credibility:.*?(?=\s*###)"
    match_credibility_to_final_answer = re.search(pattern_credibility_to_final_answer, text, re.IGNORECASE)
    
    if match_credibility_to_final_answer:
        # 如果找到 "### Credibility:" 到 "final answer" 之间的内容，去掉 "### Credibility:"
        return match_credibility_to_final_answer.group(0).split('### credibility:',1)[1].strip()
    
    # 如果没有找到 "### Credibility:" 到 "final answer"，再尝试查找 "### end" 之前的内容，兼容大小写
    pattern_end = r"(.*?)(?=\s*### end)"
    match_end = re.search(pattern_end, text, re.IGNORECASE)
    
    if match_end:
        # 如果找到 "### end"，返回其之前的内容
        return match_end.group(1).strip()
    
    # 如果没有找到 "final answer" 和 "### end"，返回整个文本
    return text.strip()


def extract_final_verdict(text):
    pattern = r"\b(yes|no)\b"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    else:
        return "" 
    

def extract_final_answer(text):
    pattern = r"final answer\s*(.*)"
    match = re.search(pattern, text)
    if match:
        answer =  match.group(1).strip()
    else:
        answer = ""
    return answer


# 生成分析的提示 (prompt)
def generate_analysis_prompt(question: str, sentence: str, media_background: str) -> str:
    prompt = f"""
### Instructions:

1. Comprehend the Question:
   - Carefully read the question to understand what is being asserted.
   - Identify the key components and assertions within the question.

2. Analyze the Sentence:
   - Examine the sentence to see how it relates to the question.
   - Determine if the sentence provides evidence, an example, or a counterpoint to the question.
   - Look for keywords or phrases that directly support or refute the question.

3. Evaluate the Media Background:
   - Review the media background information to understand the broader context.
   - Consider the credibility of the sources mentioned and any potential biases.
   - Identify any historical information or prior events that relate to the question.

4. Integrate Information:
   - Combine insights from the sentence and media background.
   - Assess whether the sentence, in the context of the media background, provides sufficient support for the question.
   - Consider if there are contradictions or alignments between the sentence and the media background.

5. Logical Reasoning:
   - Use critical thinking to evaluate the connections.
   - Ask yourself if the evidence logically leads to the conclusion stated in the question.
   - Consider alternative interpretations or whether additional information is needed.

6. Conclude:
   - Evaluate the reliability of the media background and determine whether the sentence supports the question.
   - Ensure that your conclusion is based solely on the information provided.

7. Answer:
   - Optionally, provide a justification based on the above steps, explaining your reasoning. Keep your justification under 300 words.
   - Provide a clear and concise "Yes" or "No" answer to the question.


### Question: {question}  

### Sentence: {sentence}  

### Media Background Analysis: {media_background}

"""
    prompt += """Based on the provided sentence and the media background, begin by thoroughly analyzing the evidence, giving special attention to the credibility and potential biases of the media source. After your analysis, provide your final answer to the question. Start your answer with '### Final Answer:'. Your final answer should be either 'yes' or 'no'. No additional content should follow your final answer.

    ### Example Response Format: 
    ### Analysis: [A short but clear Analysis]
    ### Credibility: [ A brief analysis]
    ### Final Answer: Yes/No *** END

    ### Analysis:"""

    return prompt



# 输入是 n 个 claim，每个 claim 对应多个 sentences 和 media background
def analyze_sentences(questions: List[str], sentences: List[List[str]], media_backgrounds: List[List[str]]) -> List[List[str]]:
    analysis_inputs = []
    for i in range(len(questions)):
        question = questions[i]
        related_sentences = sentences[i]
        related_media_backgrounds = media_backgrounds[i]

        for j in range(len(related_sentences)):
            sentence = related_sentences[j]
            media_background = related_media_backgrounds[j]
            analysis_input = generate_analysis_prompt(question, sentence, media_background)
            analysis_inputs.append(analysis_input)

    # 使用 response_batch 函数生成批量输出
    flat_analysis_results = response_batch(analysis_inputs,set_max_tokens = 500)

    reshaped_analysis_results = []
    index = 0
    for i in range(len(questions)):
        related_sentences = sentences[i]
        result_for_question = []
        for _ in related_sentences:
            result_for_question.append(flat_analysis_results[index])
            index += 1
        reshaped_analysis_results.append(result_for_question)
    
    return reshaped_analysis_results
    # return analysis_inputs, flat_analysis_results



def response_batch(input_texts: List[str], set_max_tokens = 5000) -> List[str]:
    sampling_params = SamplingParams(temperature=0.90, top_p=0.9, max_tokens = set_max_tokens, seed=args.seed, stop = '*** END')

    outputs = LLM.generate(input_texts, sampling_params)

    # Since the output from vllm may contain more information, we need to extract the generated text
    result_texts = [
    ' '.join(output.outputs[0].text.strip().lower().split()) 
    for output in outputs
    ]


    return result_texts
# 将每个 claim 的支持和反对的句子分为两类
def categorize_sentences(questions: List[str], sentences: List[List[str]], media_backgrounds: List[List[str]], analysis_results: List[List[str]]) -> List[Dict[str, List[str]]]:
    categorized_results = []

    for i in range(len(questions)):
        question = questions[i]
        related_sentences = sentences[i]
        related_media_backgrounds = media_backgrounds[i]
        support_sentences = defaultdict(list)
        oppose_sentences = defaultdict(list)

        for j in range(len(related_sentences)):
            sentence = related_sentences[j]
            media_background = related_media_backgrounds[j]
            analysis_result = analysis_results[i][j]

            verdict = extract_final_verdict(extract_final_answer(analysis_result.lower()))
            analysis_result  = extract_before_final_answer_or_end(analysis_result)   
            if "yes"  in verdict:
                support_sentences[media_background].append((sentence, analysis_result))
            elif "no" in verdict:
                oppose_sentences[media_background].append((sentence, analysis_result))
            else:
                print("no conclusion for ", sentence)
          

        categorized_results.append({
            "question": question,
            "support": support_sentences,
            "oppose": oppose_sentences
        })
    
    return categorized_results

# 使用 LLM 总结支持和反对的证据，并给出最后判断
def summarize_questions_llm(categorized_results: List[Dict[str, List[str]]]) -> List[str]:

    summary_inputs = []
    for result in categorized_results:

        summary_input = prompt_summary_analysis

        question = result["question"]
        support = result["support"]
        oppose = result["oppose"]

        summary_input += f"### Question: {question}\n\n### Support Evidence:\n"
        if not support:
            summary_input += "No support evidence found.\n"
        for background, sentences in support.items():
            for sentence, analysis in sentences:
                summary_input += f"- Sentence: {sentence}\n- Credibility Analysis: {analysis}\n\n"
                #summary_input += f"- Sentence: {sentence}\n"

        summary_input += "\n### Oppose Evidence:\n"
        if not oppose:
            summary_input += "No oppose evidence found.\n"
        for background, sentences in oppose.items():
            #summary_input += f"Media Background: {background}\n"
            for sentence, analysis in sentences:
                summary_input += f"- Sentence: {sentence}\n- Credibility Analysis: {analysis}\n\n"
                # summary_input += f"- Sentence: {sentence}\n"

        summary_input += (
            "\nGiven the above support and oppose evidence, first explain your reasoning for any contradictions or conflicting information. Your analysis should be no more than 500 words. Please ignore the difference in the amount of supporting and opposing evidence and choose more detailed and truthful sentences of evidence."
            "Once you have completed your analysis, provide your final answer to the question based on the evidence you analyzed. "
            "Start your answer with '### Final Answer:' and ensure it is clearly separated from your evidence analysis."
            "Your final answer should be either 'yes' or 'no'."
            "Make sure to include only one final answer, and do not include any additional text after it.\n\n"
            "### Example Response Format: \n"
            "### Analysis: [In-depth analysis]\n"
            "### Final Answer: [yes/no] *** END\n\n"
            "### Your Response: \n"
            #"### Analysis: "
        )

        summary_inputs.append(summary_input)
    
    # 使用 LLM 生成总结和最终判断
    summaries = response_batch(summary_inputs, set_max_tokens = 5000)

    return summaries

# 进一步处理 questions 和 sentences
def process_questions(questions, sentences, media_backgrounds, output_file) -> List[str]:

    analysis_results = analyze_sentences(questions, sentences, media_backgrounds)
    torch.cuda.empty_cache()
    categorized_results = categorize_sentences(questions, sentences, media_backgrounds, analysis_results)
    summaries = summarize_questions_llm(categorized_results)

    final_answers = []
    for idx, output in enumerate(summaries):
        results = f"Question: {questions[idx]}\n\n"
        final_answer = extract_final_answer(output)
        final_answers.append(final_answer)
        results += f"Final answer: {final_answer}\n\n"
        results += f"output: {output}\n"
        results += "\n\n-----------------------------------------------------\n\n"
        write_to_file(output_file, results)

    return final_answers