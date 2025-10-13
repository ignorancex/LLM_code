SYSTEM_PROMPT_STANDARD = (
    "You are a helpful AI assistant skilled in extractive question answering."
)
SYSTEM_PROMPT_ENGLISH = (
    """You are a helpful AI assistant skilled in question answering."""
)

system_prompt_dict = {
    "standard": SYSTEM_PROMPT_STANDARD,
    "english": SYSTEM_PROMPT_ENGLISH,
}

QUERY_TEMPLATE = """Given the following passage and question, extract the exact answer from the passage. The answer should be a short span of text found verbatim in the passage.
###
Passage:
{passage}
###
Question:
{question}
###
Answer:"""

ENG_QUERY_TEMPLATE = """Given the following passage and question, return the answer in English using only the information from the passage. The answer should be a concise response based on the provided content.
###
Passage:
{passage}
###
Question:
{question}
###
Answer:"""

ENG_DETAILED_TEMPLATE = """Given a passage written in a non-English language, followed by a question written in English. Your task is to extract the answer from the passage and provide it in English. If the answer is not explicitly mentioned in the passage, respond with "The passage does not provide this information."
###
Passage:
{passage}
###
Question:
{question}
###
Answer:"""


template_dict = {
    "standard": QUERY_TEMPLATE,
    "eng-standard": ENG_QUERY_TEMPLATE,
    "eng-detailed": ENG_DETAILED_TEMPLATE,
}
