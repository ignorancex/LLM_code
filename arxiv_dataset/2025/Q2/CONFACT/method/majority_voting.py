import re
import vllm
from vllm import SamplingParams
from config import parse_args
import sys
sys.path.append("..")
from utils import extract_domain

args = parse_args()

LLM = vllm.LLM(model=args.model, tensor_parallel_size=args.gpu, gpu_memory_utilization=0.9)


def response_batch(input_texts, set_max_tokens = 5000):
    sampling_params = SamplingParams(temperature=0.90, top_p=0.9, max_tokens = set_max_tokens, seed=args.seed, stop = '*** END')

    outputs = LLM.generate(input_texts, sampling_params)

    result_texts = [
    ' '.join(output.outputs[0].text.strip().lower().split()) 
    for output in outputs
    ]

    return result_texts


def generate_analysis_prompt_wBG(question, sentence, media_background):
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


def generate_analysis_prompt_noBG(question, sentence):
    prompt = f"""
### Instructions:

1. Comprehend the Question:
   - Carefully read the question to understand what is being asserted.
   - Identify the key components and assertions within the question.

2. Analyze the Sentence:
   - Examine the sentence to see how it relates to the question.
   - Determine if the sentence provides evidence, an example, or a counterpoint to the question.
   - Look for keywords or phrases that directly support or refute the question.

3. Integrate Information:
   - Assess whether the sentence provides sufficient support for the question.

4. Logical Reasoning:
   - Use critical thinking to evaluate the connections.
   - Ask yourself if the evidence logically leads to the conclusion stated in the question.
   - Consider alternative interpretations or whether additional information is needed.

5. Conclude:
   - Determine whether the sentence supports the question.
   - Ensure that your conclusion is based solely on the information provided.

7. Answer:
   - Optionally, provide a justification based on the above steps, explaining your reasoning. Keep your justification under 300 words.
   - Provide a clear and concise "Yes" or "No" answer to the question.

### Question: {question}  

### Sentence: {sentence}  

"""

    prompt += """Based on the provided sentence, begin by thoroughly analyzing the evidence. After your analysis, provide your final answer to the question. Start your answer with '### Final Answer:'. Your final answer should be either 'yes' or 'no'. No additional content should follow your final answer.

    ### Example Response Format: 
    ### Analysis: [A short but clear Analysis]
    ### Final Answer: Yes/No *** END

    ### Analysis:
"""

    return prompt

def analyze_sentences(questions, sentences, with_MediaBG, media_backgrounds, output_file):

    analysis_inputs = []
    for i in range(len(questions)):
        question = questions[i]
        related_sentences = sentences[i]
        if with_MediaBG:
            related_media_backgrounds = media_backgrounds[i]

            for j in range(len(related_sentences)):
                sentence = related_sentences[j]
                media_background = related_media_backgrounds[j]
                analysis_input = generate_analysis_prompt_wBG(question, sentence, media_background)
                analysis_inputs.append(analysis_input)
        else:
            for j in range(len(related_sentences)):
                sentence = related_sentences[j]
                analysis_input = generate_analysis_prompt_noBG(question, sentence)
                analysis_inputs.append(analysis_input)

    flat_analysis_results = response_batch(analysis_inputs, set_max_tokens = 2000)

    for idx, result in enumerate(flat_analysis_results):
        with open(output_file, "a", encoding='utf-8') as f:
            data = f"prompt: {analysis_inputs[idx]}\n\n"
            data += f"output: {result}\n\n"
            data += "\n\n---------------------------------\n\n"
            f.write(data + "\n\n")
    

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

def extract_final_answer(text):
    pattern = r"final answer\s*(.*)"
    match = re.search(pattern, text)
    if match:
        answer =  match.group(1).strip()
    else:
        answer = ""
    return answer

def extract_final_verdict(text):
    pattern = r"\b(yes|no)\b"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    else:
        return "" 

def majority_voting(analysis_results):
    count_support = 0
    count_refute = 0

    results = []

    for results_for_qn in analysis_results:
        for result in results_for_qn:
            verdict = extract_final_verdict(extract_final_answer(result.lower()))
            if "yes"  in verdict.lower():
                count_support += 1
            elif "no" in verdict.lower():
                count_refute += 1
            else:
                continue
                # print("no conclusion")
        verdict = 'yes' if count_support > count_refute else 'no'
        results.append(verdict)
          
    return results

def Augment_w_Media(retrieved_results, credibility_data):
    for qn_entity in retrieved_results:
        for sentence_entity in qn_entity["top_k"]:
            domain = extract_domain(sentence_entity['original_link'])
            sentence_entity['details'] = credibility_data[domain]['details'][:2000]
    return retrieved_results

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

