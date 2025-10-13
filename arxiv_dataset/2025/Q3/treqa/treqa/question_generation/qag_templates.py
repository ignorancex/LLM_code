SYSTEM_PROMPT_STANDARD = "You are a helpful AI assistant skilled in generating questions and answers from given passages."
SYSTEM_PROMPT_TEACHER = """You are an expert teacher who is tasked with designing reading comprehension questions from a text."""

system_prompt_dict = {
    "standard": SYSTEM_PROMPT_STANDARD,
    "teacher": SYSTEM_PROMPT_TEACHER,
}

QUERY_TEMPLATE = """Generate {num_questions} question-answer pairs based on the following passage. The questions should be diverse and cover different aspects of the passage. Each question should be answerable using information from the passage, and the answer must be a direct extract from the passage. Provide the exact text from the passage as the answer. Answer in the format `Q: <question>\nA: <answer>\n\n`....

Passage:
{passage}

Question-Answer Pairs:
"""

QUERY_DETAILED_TEMPLATE = """Generate {num_questions} question-answer pairs based on the following passage. These questions should be specifically designed to evaluate the quality of machine translation systems. Focus on the following aspects:

1. Semantic accuracy: Questions that test if key meanings and concepts are preserved.
2. Terminology: Questions about specific terms or domain-specific vocabulary.
3. Idiomatic expressions: Questions targeting idiomatic phrases or culturally specific references.
4. Grammatical structure: Questions that check for correct handling of complex sentence structures.
5. Named entities: Questions about proper names, locations, or organizations.

Each question should be answerable using information from the passage, and the answer must be a direct extract from the passage. Provide the exact text from the passage as the answer. Answer in the format `Q: <question>\nA: <answer>\n\n`.

Passage:
{passage}

Question-Answer Pairs:
"""

QUERY_ANSWER_FIRST_TEMPLATE = """Read the given text carefully and identify short (1-{num_questions} words) key phrases, concepts, or entities that are important in the context. Based on the key phrases, generate relevant question-answer pairs that test the reader's understanding of the information in the passage. Use the exact text from the passage as the answer. Answer in the format

Q: <question1>
A: <answer1>

Q: <question2>
A: <answer2>

...

Passage:
{passage}

Question-Answer Pairs:
"""

QUERY_ANSWER_FIRST_FIXED_TEMPLATE = """Read the given text carefully and identify short (1-5 words) key phrases, concepts, or entities that are important in the context. Based on the key phrases, generate {num_questions} relevant question-answer pairs that test the reader's understanding of the information in the passage. Use the exact text from the passage as the answer. Answer in the format

Q: <question1>
A: <answer1>

Q: <question2>
A: <answer2>

...

Passage:
{passage}

Question-Answer Pairs:
"""


ENG_QUERY_TEMPLATE = """Generate {num_questions} question-answer pairs based on the following passage. Each question should be answerable using information from the passage. The questions and answers must be strictly in English, while ensuring that the meaning of the answer is preserved. The questions should be diverse and cover different aspects of the passage. Answer in the format `Q: <question>\nA: <answer>\n\n`....

Passage:
{passage}

Question-Answer Pairs:
"""

ENG_QUERY_DETAILED_TEMPLATE = """Generate {num_questions} question-answer pairs based on the following passage. These questions should be specifically designed to evaluate the quality of machine translation systems. Focus on the following aspects:

1. Semantic accuracy: Questions that test if key meanings and concepts are preserved.
2. Terminology: Questions about specific terms or domain-specific vocabulary.
3. Idiomatic expressions: Questions targeting idiomatic phrases or culturally specific references.
4. Grammatical structure: Questions that check for correct handling of complex sentence structures.
5. Named entities: Questions about proper names, locations, or organizations.

Each question should be answerable using information from the passage. Provide the questions and answers strictly in English, while ensuring that the meaning of the answer is preserved. Answer in the format `Q: <question>\nA: <answer>\n\n`.

Passage:
{passage}

Question-Answer Pairs:
"""

ENG_QUERY_ANSWER_FIRST_TEMPLATE = """Read the given text carefully and identify short (1-{num_questions} words) key phrases, concepts, or entities that are important in the context. Based on the key phrases, generate relevant question-answer pairs that test the reader's understanding of the information in the passage. Provide the questions and answers strictly in English, while ensuring that the meaning of the answer is preserved. Answer in the format

Q: <question1>
A: <answer1>

Q: <question2>
A: <answer2>

...

Passage:
{passage}

Question-Answer Pairs:
"""

ENG_QUERY_ANSWER_FIRST_TEMPLATE_1SHOT = """Read the given text carefully and identify short (1-5 words) key phrases, concepts, or entities that are important in the context. Based on the key phrases, generate relevant question-answer pairs that test the reader's understanding of the information in the passage. Provide the questions and answers strictly in English, while ensuring that the meaning of the answer is preserved.

Here's an example:

Passage:
我并不打算赴约。不管是因为什么，我想我们都没有再见面的必要了。我坐在沙发上一支一支地抽烟，天色越来越暗，门突然笃笃地敲响了。送水的男孩扛着水桶站在门囗，说是给西郊的一户人家送水去了。他戴着一顶脏兮兮的灰色毛线帽子，神情恍惚。

Question-Answer Pairs:
Q: Who knocked at the door?
A: The water delivery boy

Q: What was the person wearing?
A: A dirty gray woolen hat

Q: Where was he delivering water to?
A: A family in the western suburbs

Now generate similar question-answer pairs for:

Passage:
{passage}

Question-Answer Pairs:
"""

ENG_QUERY_ANSWER_FIRST_FIXED_TEMPLATE = """Read the given text carefully and identify short (1-5 words) key phrases, concepts, or entities that are important in the context. Based on the key phrases, generate {num_questions} relevant question-answer pairs that test the reader's understanding of the information in the passage. Provide the questions and answers strictly in English, while ensuring that the meaning of the answer is preserved. Answer in the format

Q: <question1>
A: <answer1>

Q: <question2>
A: <answer2>

...

Passage:
{passage}

Question-Answer Pairs:
"""

ENG_QUERY_NOCANDS_TEMPLATE = """"Generate{num_questions} question-answer pairs to verify translation accuracy. Each answer should be a key phrase, concept, or entity from the original passage that are important in the context and could help detect potential errors or mistranslations.
The questions and answers must be strictly in English, while ensuring that the meaning of the answer is preserved. The questions should be diverse and cover different aspects of the passage. Answer in the format:

Q: <question1>
A: <answer1>

Q: <question2>
A: <answer2>

...

Original Passage:
{passage}

Question-Answer Pairs:
"""

ENG_QUERY_NOCANDS_1SHOT_TEMPLATE = """"Generate{num_questions} question-answer pairs to verify translation accuracy. Each answer should be a key phrase, concept, or entity from the original passage that are important in the context and could help detect potential errors or mistranslations.
The questions and answers must be strictly in English, while ensuring that the meaning of the answer is preserved. The questions should be diverse and cover different aspects of the passage. Answer in the format:

Q: <question1>
A: <answer1>

Q: <question2>
A: <answer2>

...

Here's an example:

Original Passage:
I hadn't planned to show up—I didn't think we had any need to meet again. I sat on the couch and smoked cigarette after cigarette as it got dark. Then someone tapped on the door: the water delivery boy. He wore a filthy gray wool hat and looked harried. "I had to make a delivery on the west side," he said.

Question-Answer Pairs:
Q: Who tapped on the door?
A: the water delivery boy

Q: What did the person wear?
A: filthy gray wool hat

Now, generate similar question-answer pairs for the following passage:

Original Passage:
{passage}

Question-Answer Pairs:
"""

template_dict = {
    "detailed": QUERY_DETAILED_TEMPLATE,
    "answer-first": QUERY_ANSWER_FIRST_TEMPLATE,
    "answer-first-fixed": QUERY_ANSWER_FIRST_FIXED_TEMPLATE,
    "standard": QUERY_TEMPLATE,
    "eng-detailed": ENG_QUERY_DETAILED_TEMPLATE,
    "eng-answer-first": ENG_QUERY_ANSWER_FIRST_TEMPLATE,
    "eng-answer-first-fixed": ENG_QUERY_ANSWER_FIRST_FIXED_TEMPLATE,
    "eng-plain": ENG_QUERY_TEMPLATE,
    # this is the one we use
    "eng-nocands": ENG_QUERY_NOCANDS_TEMPLATE,
    "eng-nocands-1shot": ENG_QUERY_NOCANDS_1SHOT_TEMPLATE,
}

# --- discriminative, candidate-based templates ---

QUERY_DISCRIMINATIVE_TEMPLATE = """"Generate{num_questions} question-answer pairs to verify translation accuracy. Each answer should be a key phrase, concept, or entity from the original passage that could help detect errors or mistranslations in the alternatives.
Each answer must be a direct quote from the original passage. Answer in the format:

Q: <question1>
A: <answer1>

Q: <question2>
A: <answer2>

...

Original Passage:
{passage}

Alternative Passages:
{alternatives}

Question-Answer Pairs:
"""

QUERY_DISCRIMINATIVE_ANSWER_FIRST_TEMPLATE = """Read the original passage and its alternatives carefully. First identify short (1-{num_questions} words) key phrases, concepts, or entities from the original passage that could help detect errors or mistranslations in the alternatives. Based on these key phrases, generate relevant question-answer pairs.

Each answer must be a direct quote from the original passage. Answer in the format:

Q: <question1>
A: <answer1>

Q: <question2>
A: <answer2>

...

Original Passage:
{passage}

Alternative Passages:
{alternatives}

Question-Answer Pairs:
"""

QUERY_DISCRIMINATIVE_ANSWER_FIRST_TEMPLATE_1SHOT = """Read the original passage and its alternatives carefully. First identify short (1-{num_questions} words) key phrases, concepts, or entities from the original passage that could help detect errors or mistranslations in the alternatives. Based on these key phrases, generate relevant question-answer pairs.

Each answer must be a direct quote from the original passage.

Here's an example:

Original Passage:
I hadn't planned to show up—I didn't think we had any need to meet again. I sat on the couch and smoked cigarette after cigarette as it got dark. Then someone tapped on the door: the water delivery boy. He wore a filthy gray wool hat and looked harried. "I had to make a delivery on the west side," he said.

Alternative Passages:
1. I had no intention of keeping the appointment. For whatever reason, I felt that there was no longer any need for us to meet. I was sitting on the sofa, smoking one cigarette after another, when the sky gradually darkened. Suddenly, there was a knock at the door. The water delivery boy was standing outside, carrying a bucket of water, saying he was delivering it to a family in the western suburbs. He wore a dirty gray woolen hat, and his expression was dazed.
2. I had no intention of keeping the appointment. No matter the reason, I don't think there's any need for us to meet again. I was sitting on the sofa, smoking one cigarette after another, when the door suddenly creaked open. The sky was getting darker and darker. The water delivery boy stood at the gate with a bucket of water, saying he was going to deliver it to a family in the western suburbs. He wore a dirty gray woolen cap and had a dazed expression.

Question-Answer Pairs:
Q: Who tapped on the door?
A: the water delivery boy

Now generate similar question-answer pairs for:

Original Passage:
{passage}

Alternative Passages:
{alternatives}

Question-Answer Pairs:
"""

QUERY_DISCRIMINATIVE_ANSWER_FIRST_TEMPLATE_ENG = """Read the original passage and its alternatives carefully. First identify short (1-{num_questions} words) key phrases, concepts, or entities from the original passage that could help detect errors or mistranslations in the alternatives. Based on these key phrases, generate relevant question-answer pairs strictly in English.

The questions and answers must be strictly in English, while ensuring that the meaning of the answer is preserved. The questions should be diverse and cover different aspects of the passage. Answer in the format:

Q: <question1>
A: <answer1>

Q: <question2>
A: <answer2>

...

Original Passage:
{passage}

Alternative Passages:
{alternatives}

Question-Answer Pairs:
"""

QUERY_DISCRIMINATIVE_ANSWER_FIRST_FIXED_TEMPLATE = """Read the original passage and its alternatives carefully. First identify short (1-5 words) key phrases, concepts, or entities from the original passage that could help detect errors or mistranslations in the alternatives. Based on these key phrases, generate {num_questions} relevant question-answer pairs strictly in English.

Each answer must be a direct quote from the original passage. Answer in the format:

Q: <question1>
A: <answer1>

Q: <question2>
A: <answer2>

...

Original Passage:
{passage}

Alternative Passages:
{alternatives}

Question-Answer Pairs:
"""


ENG_QUERY_DISCRIMINATIVE_TEMPLATE = """"Generate{num_questions} question-answer pairs to verify translation accuracy. Each answer should be a key phrase, concept, or entity from the original passage that could help detect errors or mistranslations in the alternatives.
The questions and answers must be strictly in English, while ensuring that the meaning of the answer is preserved. The questions should be diverse and cover different aspects of the passage. Answer in the format:

Q: <question1>
A: <answer1>

Q: <question2>
A: <answer2>

...

Original Passage:
{passage}

Alternative Passages:
{alternatives}

Question-Answer Pairs:
"""

ENG_QUERY_DISCRIMINATIVE_1SHOT_TEMPLATE = """"Generate{num_questions} question-answer pairs to verify translation accuracy. Each answer should be a key phrase, concept, or entity from the original passage that could help detect errors or mistranslations in the alternatives.
The questions and answers must be strictly in English, while ensuring that the meaning of the answer is preserved. The questions should be diverse and cover different aspects of the passage. Answer in the format:

Q: <question1>
A: <answer1>

Q: <question2>
A: <answer2>

...

Here's an example:

Original Passage:
我并不打算赴约。不管是因为什么，我想我们都没有再见面的必要了。我坐在沙发上一支一支地抽烟，天色越来越暗，门突然笃笃地敲响了。送水的男孩扛着水桶站在门囗，说是给西郊的一户人家送水去了。他戴着一顶脏兮兮的灰色毛线帽子，神情恍惚。

Alternative Passages:
I had no intention of keeping the appointment. For whatever reason, I felt that there was no longer any need for us to meet. I was sitting on the sofa, smoking one cigarette after another, when the sky gradually darkened. Suddenly, there was a knock at the door. The water delivery boy was standing outside, carrying a bucket of water, saying he was delivering it to a family in the western suburbs. He wore a dirty gray woolen hat, and his expression was dazed.
-
I had no intention of keeping the appointment. No matter the reason, I don't think there's any need for us to meet again. I was sitting on the sofa, smoking one cigarette after another, when the door suddenly creaked open. The sky was getting darker and darker. The water delivery boy stood at the gate with a bucket of water, saying he was going to deliver it to a family in the western suburbs. He wore a dirty gray woolen cap and had a dazed expression.

Question-Answer Pairs:
Q: Who tapped on the door?
A: the water delivery boy

Now generate similar question-answer pairs for:

Original Passage:
{passage}

Alternative Passages:
{alternatives}

Question-Answer Pairs:
"""


disc_template_dict = {
    "standard": QUERY_DISCRIMINATIVE_TEMPLATE,
    "answer-first": QUERY_DISCRIMINATIVE_ANSWER_FIRST_TEMPLATE,
    "answer-first-1shot": QUERY_DISCRIMINATIVE_ANSWER_FIRST_TEMPLATE_1SHOT,
    "eng-answer-first": QUERY_DISCRIMINATIVE_ANSWER_FIRST_TEMPLATE_ENG,
    "answer-first-fixed": QUERY_DISCRIMINATIVE_ANSWER_FIRST_FIXED_TEMPLATE,
    # this is the one we use
    "eng-cands": ENG_QUERY_DISCRIMINATIVE_TEMPLATE,
    "eng-cands-1shot": ENG_QUERY_NOCANDS_1SHOT_TEMPLATE,
}


# --- Prompts that use both source ans reference ---

ENG_QUERY_EVALUATION_ZEROSHOT_TEMPLATE = """"Generate{num_questions} question-answer pairs to verify translation accuracy. Each answer should be a key phrase, concept, or entity from the original passage (source or reference) that could help detect errors or mistranslations in the candidate(s).
The questions and answers must be strictly in English, while ensuring that the meaning of the answer is preserved. The questions should be diverse and cover different aspects of the passage. Answer in the format:

Q: <question1>
A: <answer1>

Q: <question2>
A: <answer2>

...

Source Passage:
{src_passage}

Reference Passage:
{ref_passage}

Candidate Passage(s):
{alternatives}

Question-Answer Pairs:
"""

ENG_QUERY_EVALUATION_1SHOT_TEMPLATE = """"Generate{num_questions} question-answer pairs to verify translation accuracy. Each answer should be a key phrase, concept, or entity from the original passage (source or reference) that could help detect errors or mistranslations in the candidate(s).
The questions and answers must be strictly in English, while ensuring that the meaning of the answer is preserved. The questions should be diverse and cover different aspects of the passage. Answer in the format:

Q: <question1>
A: <answer1>

Q: <question2>
A: <answer2>

...

Here's an example:

Source Passage:
我并不打算赴约。不管是因为什么，我想我们都没有再见面的必要了。我坐在沙发上一支一支地抽烟，天色越来越暗，门突然笃笃地敲响了。送水的男孩扛着水桶站在门囗，说是给西郊的一户人家送水去了。他戴着一顶脏兮兮的灰色毛线帽子，神情恍惚。

Reference Passage:
I hadn't planned to show up—I didn't think we had any need to meet again. I sat on the couch and smoked cigarette after cigarette as it got dark. Then someone tapped on the door: the water delivery boy. He wore a filthy gray wool hat and looked harried. "I had to make a delivery on the west side," he said.

Candidate Passage(s):
I had no intention of keeping the appointment. For whatever reason, I felt that there was no longer any need for us to meet. I was sitting on the sofa, smoking one cigarette after another, when the sky gradually darkened. Suddenly, there was a knock at the door. The water delivery boy was standing outside, carrying a bucket of water, saying he was delivering it to a family in the western suburbs. He wore a dirty gray woolen hat, and his expression was dazed.
-
I had no intention of keeping the appointment. No matter the reason, I don't think there's any need for us to meet again. I was sitting on the sofa, smoking one cigarette after another, when the door suddenly creaked open. The sky was getting darker and darker. The water delivery boy stood at the gate with a bucket of water, saying he was going to deliver it to a family in the western suburbs. He wore a dirty gray woolen cap and had a dazed expression.

Question-Answer Pairs:
Q: Who knocked at the door?
A: The water delivery boy

Now generate similar question-answer pairs for:

Source Passage:
{src_passage}

Reference Passage:
{ref_passage}

Candidate Passage(s):
{alternatives}

Question-Answer Pairs:
"""

eval_template_dict = {
    "eng-cands-0shot": ENG_QUERY_EVALUATION_ZEROSHOT_TEMPLATE,
    "eng-cands-1shot": ENG_QUERY_EVALUATION_1SHOT_TEMPLATE,
}

ENG_QUERY_NOCANDS_BOTH_TEMPLATE = """"Generate{num_questions} question-answer pairs to verify translation accuracy. Each answer should be a key phrase, concept, or entity from the original passage (source or reference) that are important in the context and could help detect potential errors or mistranslations.
The questions and answers must be strictly in English, while ensuring that the meaning of the answer is preserved. The questions should be diverse and cover different aspects of the passage. Answer in the format:

Q: <question1>
A: <answer1>

Q: <question2>
A: <answer2>

...

Source Passage:
{src_passage}

Reference Passage:
{ref_passage}

Question-Answer Pairs:
"""

template_dict_both = {
    "eng-both-nocands-0shot": ENG_QUERY_NOCANDS_BOTH_TEMPLATE,
}
