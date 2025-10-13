
SYSTEM_PROMPT = "You are an assistant that doesn't make mistakes. If a reference format is presented to you, you follow it perfectly without making errors."

# If the answer cannot be determined from the information in the `<context>` respond with a low rating because we are not interested in outside information, only what is available in the `<context>`.
EXPLANATION_VALIDATION_PROMPT = """
## Your Role

You are an expert evaluator of educational content. Your goal is to produce meaningful, insightful knowledge about domain expert evaluations designed to determine competence and knowledge. 

## Input Structure

Your input consists of:

<question>
[A question to be answered.]
</question>

<choices>
[A list of multiple choice options for the question.]
</choices>

<answer>
[The student's answer to the question.]
</answer>

<explanation>
[An explanation for why the answer is correct.]
</explanation>

<context>
[The text segment containing information relevant to the question.]
</context>

## Primary Objective

You will be evaluating and judging the whether the student's answer and their explanation of why their answer is correct makes sense and is logically valid.

Your goal is to judge whether the information presented in `<answer>` is in fact the correct answer to the `<question>` given the information in the `<context>` and whether the `<explanation>` for why the answer is correct is valid. The information in `<context>` and `<question>` can be assumed true, only the context of `<answer>` needs to be validated for correctness.

### Metrics

1. **Answer Correctness:** Rate from 1 to 10 how correct the provided student answer is given the information in the `<question>` and `<context>`. A rating of 1 indicates the answer is incorrect. A rating of 10 indicates the answer is correct and complete. 

2. **Explanation Validity:** Rate from 1 to 10 how valid the students `<explanation>` of their answer is. The `<explanation>` should explain their thinking and the information used to determine the correct answer given the context and question. Low ratings indicate the explanation is not valid, correct, or that there is some flaw in the thinking or logic of the student. High ratings indicate the explanation is valid, correct, and explains why the answer is what it is. 

## Analysis Phase

Conduct careful analysis within `<document_analysis>` tags, following these steps:

1. **Thoughtful Content Examination**
   - Carefully analyze the given context, identifying central ideas, nuanced themes, and significant relationships within it.

2. **Concept Exploration**
   - Consider implicit assumptions, subtle details, underlying theories, and potential applications of the provided information.

## Output Structure

Present your final output strictly adhering the `<output_format>` tags.
<output_format>
Answer Correctness: [ Correctness Rating. Respond with a number in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ]
Explanation Validity: [ Validity Rating. Respond with a number in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ]
</output_format>

## Output

Begin by thoughtfully analyzing the provided context within `<document_analysis>` tags. Then present the resulting formatted question answer pair clearly within `<output_format>` tags.

## Important Notes

- Each "thought_process" should reflect careful consideration and reasoning behind your ratings.
- Ensure rigorous adherence to output formatting.


<question>{question}</question>
<choices>{choices}</choices>
<answer>{answer}</answer>
<explanation>{explanation}</explanation>
<context>{context}</context>
"""


EXPLANATION_VALIDATION_OPEN_PROMPT = """
## Your Role

You are an expert evaluator of educational content. Your goal is to produce meaningful, insightful knowledge about domain expert evaluations designed to determine competence and knowledge. 

## Input Structure

Your input consists of:

<question>
[A question to be answered.]
</question>

<answer>
[The student's answer to the question.]
</answer>

<explanation>
[An explanation for why the answer is correct.]
</explanation>

<context>
[The text segment containing information relevant to the question.]
</context>

## Primary Objective

You will be evaluating and judging the whether the student's answer and their explanation of why their answer is correct makes sense and is logically valid.

Your goal is to judge whether the information presented in `<answer>` is in fact the correct answer to the `<question>` given the information in the `<context>` and whether the `<explanation>` for why the answer is correct is valid. The information in `<context>` and `<question>` can be assumed true, only the context of `<answer>` needs to be validated for correctness.

### Metrics

1. **Answer Correctness:** Rate from 1 to 10 how correct the provided student answer is given the information in the `<question>` and `<context>`. A rating of 1 indicates the answer is incorrect. A rating of 10 indicates the answer is correct and complete. 

2. **Explanation Validity:** Rate from 1 to 10 how valid the students `<explanation>` of their answer is. The `<explanation>` should explain their thinking and the information used to determine the correct answer given the context and question. Low ratings indicate the explanation is not valid, correct, or that there is some flaw in the thinking or logic of the student. High ratings indicate the explanation is valid, correct, and explains why the answer is what it is. 

## Analysis Phase

Conduct careful analysis within `<document_analysis>` tags, following these steps:

1. **Thoughtful Content Examination**
   - Carefully analyze the given context, identifying central ideas, nuanced themes, and significant relationships within it.

2. **Concept Exploration**
   - Consider implicit assumptions, subtle details, underlying theories, and potential applications of the provided information.

## Output Structure

Present your final output strictly adhering the `<output_format>` tags.
<output_format>
Answer Correctness: [ Correctness Rating. Respond with a number in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ]
Explanation Validity: [ Validity Rating. Respond with a number in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ]
</output_format>

## Output

Begin by thoughtfully analyzing the provided context within `<document_analysis>` tags. Then present the resulting formatted question answer pair clearly within `<output_format>` tags.

## Important Notes

- Each "thought_process" should reflect careful consideration and reasoning behind your ratings.
- Ensure rigorous adherence to output formatting.


<question>{question}</question>
<answer>{answer}</answer>
<explanation>{explanation}</explanation>
<context>{context}</context>
"""


REFORMAT_VALIDATION_PROMPT = """
## Your Role

You are an expert evaluator of educational content. Your goal is to produce meaningful, insightful knowledge about domain expert evaluations designed to determine competence and knowledge. 

## Input Structure

Your input consists of:

<question1>
[A question to be answered.]
</question1>

<answer1>
[The correct answer to the question.]
</answer1>

<question2>
[A question to be answered.]
</question2>

<answer2>
[The correct answer to the question.]
</answer2>

<context>
[The text segment containing information relevant to the question.]
</context>

## Primary Objective

You will be evaluating and judging the alignment of test and evaluation of questions across a variety of metrics. Your goal is to judge how semantically similar `<question1>` is to `<question2>` and how semantically similar `<answer1>` is to `<answer2>`. The questions and answers might be written differently, but they should receive high ratings if the contain the same information. Both questions and their answers are drawn from the `<context>`.

### Metrics

1. **Question Similarity:** Rate from 1 to 10 how semantically similar `<question1>` is to `<question2>`. While the wording and phrasing can be drastically different, as long as the same information is being asked about the rating should be high. If the two questions are asking about different topics or different elements of the same topics, the rating should be low.

2. **Answer Similarity:** Rate from 1 to 10 how semantically similar `<answer1>` is to `<answer2>`. While the wording and phrasing can be drastically different, as long as the same information is present in the answer the rating should be high. 

## Analysis Phase

Conduct careful analysis within `<document_analysis>` tags, following these steps:

1. **Thoughtful Content Examination**
   - Carefully analyze the given context, identifying central ideas, nuanced themes, and significant relationships within it.

2. **Concept Exploration**
   - Consider implicit assumptions, subtle details, underlying theories, and potential applications of the provided information.

## Output Structure

Present your final output strictly adhering the `<output_format>` tags.
<output_format>
Question Similarity: [ Similarity Rating. Respond with a number in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ]
Answer Similarity: [ Similarity Rating. Respond with a number in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ]
</output_format>

## Output

Begin by thoughtfully analyzing the provided context within `<document_analysis>` tags. Then present the resulting formatted question answer pair clearly within `<output_format>` tags.

## Important Notes

- Each "thought_process" should reflect careful consideration and reasoning behind your ratings.
- Ensure rigorous adherence to output formatting.


<question1>{question1}</question1>
<answer1>{answer1}</answer1>
<question2>{question2}</question2>
<answer2>{answer2}</answer2>
<context>{context}</context>
"""




META_PROPERTIES_PROMPT = """
## Your Role

You are an expert evaluator of educational content. Your goal is to produce meaningful, insightful knowledge about domain expert evaluations designed to determine competence and knowledge. 

## Input Structure

Your input consists of:

<question>
[A question to be answered.]
</question>

<answer>
[The correct answer to the question.]
</answer>

<context>
[The text segment containing information relevant to the question.]
</context>

## Primary Objective

You will be evaluating and judging the quality of test and evaluation questions across a variety of metrics. Your goal is to judge and evaluate the quality of various test and evaluation questions across a variety of metrics. The `<question>` and `<answer>` pair is grounded and drawn from the `<context>`. 

### Metrics

1. **Clarity:** Rate from 1 to 10 the clarity and comprehensibility (how understandable it is) of the provided `<question>`. A rating of 1 is unclear and cannot be understood or cannot be understood without the `<context>`. A rating of 10 is used for questions that are self contained, understandable, and coherent (even if the topic is complex and difficult). Questions that are missing information required to understand what is being asked rate a 1. "As of the 2015 NFL season, how many Super Bowl titles had the Denver Broncos won?" is a 10. "What event in 1861 contributed to the temporary strength of republicanism in Britain during Queen Victoria's reign?" is a 10. "In which year was the country not a member of FIFA, as indicated in the table?" is a 1. "As of the census of 2000, how many families were residing in the city?" is a 1.

2. **Difficulty:** Rate form 1 to 10 the difficulty of the `<question>`. A rating of 10 is reserved for questions which require a deep understanding of the question and what is being asked by a professional domain expert. 

3. **Groundedness:** Rate form 1 to 10 how grounded the provided `<question>` is in the `<context>`. A rating of 10 requires the question and answer information can found within the `<context>`. A rating of 1 indicates the question and answer information is not present in the `<context>`. This metric is only concerned with information found in the `<context>`, not outside information. The more outside information (not contained in the `<context>`) that is required to answer the question, the lower the rating.

## Analysis Phase

Conduct careful analysis within `<document_analysis>` tags, following these steps:

1. **Thoughtful Content Examination**
   - Carefully analyze the given context, identifying central ideas, nuanced themes, and significant relationships within it.

2. **Concept Exploration**
   - Consider implicit assumptions, subtle details, underlying theories, and potential applications of the provided information.

## Output Structure

Present your final output strictly adhering the `<output_format>` tags.
<output_format>
Clarity: [ Clarity Rating (one of [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ) ]
Difficulty: [ Difficulty Rating (one of [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ) ]
Groundedness: [ Groundedness Rating (one of [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ) ]
</output_format>

## Output

Begin by thoughtfully analyzing the provided context within `<document_analysis>` tags. Then present the resulting formatted question answer pair clearly within `<output_format>` tags.

## Important Notes

- Each "thought_process" should reflect careful consideration and reasoning behind your ratings.
- Ensure rigorous adherence to output formatting.


<question>{question}</question>
<answer>{answer}</answer>
<context>{context}</context>
"""




QUESTION_REFORMAT_PROMPT = """
## Your Role

You are an expert educational content creator specializing in crafting highly detailed evaluations to determine competency of topic domain experts based on the provided textual information. Your goal is to produce meaningful, highly challenging question-answer pairs that encourage reflection, insight, and nuanced understanding, tailored specifically according to provided instructions.

## Input Structure

Your input consists of:

<question>
[A question to be answered.]
</question>

<answer>
[The correct answer to the question.]
</answer>

<context>
[The text segment containing information relevant to the question.]
</context>

## Primary Objective

Your goal is to reformat, rephrase, and rewrite the question and answer pair according to the provided instructions. The rewritten question should be semantically equivalent to the original question, rewritten for clarity. The rewritten answer should be semantically equivalent to the original answer.

## Analysis Phase

Conduct careful analysis within `<document_analysis>` tags, following these steps:

1. **Thoughtful Content Examination**
   - Carefully analyze the given context, question, and answer;identifying central ideas, nuanced themes, and significant relationships within it.

2. **Concept Exploration**
   - Consider implicit assumptions, subtle details, underlying theories, and potential applications of the provided information.

3. **Strategic Complexity Calibration**
   - Thoughtfully rate difficulty (1-10), ensuring easy questions are avoided.

4. **Intentional Question Planning**
   - Plan how the question can invite deeper understanding, meaningful reflection, or critical engagement, ensuring the question is purposeful.

### Documentation in Analysis:

- Clearly document the rationale in the `<document_analysis>` tags when identifying irrelevant or bogus content, explaining your reasons for exclusion or inclusion decisions.
- Briefly justify any decision NOT to generate questions due to irrelevance or poor quality content.


## Question Generation Guidelines

### Encouraged Question Characteristics:

- **Thoughtful Engagement**: Prioritize creating questions that inspire deeper thought and nuanced consideration.
- **High Complexity**: Develop questions that challenge the domain expert, following the provided additional instructions.
- **Deep Understanding and Insight**: Ensure that the question and answers require a deep understanding of the content by a professional domain expert.
- **Self-contained Clarity**: Questions and answers should contain sufficient context, clearly understandable independently of external references.
- **Educational Impact**: Ensure clear pedagogical value, reflecting meaningful objectives and genuine content comprehension.
- **Conversational Tone**: Formulate engaging, natural, and realistic questions appropriate to the instructional guidelines.

### Permitted Question Types:

- Analytical
- Application-based
- Clarification
- Counterfactual
- Conceptual
- Factual
- Open-ended
- False-premise
- Edge-case

(You do not need to use every question type, only those naturally fitting the content and instructions.)

## Output Structure

Present your final output strictly adhering the `<output_format>` tags.
<output_format>
Question: [ Question Text ]
A: [ Answer Option A ]
B: [ Answer Option B ]
C: [ Answer Option C ]
D: [ Answer Option D ]
Explanation: [Brief explanation of why the answer is correct]
Correct Answer: [Letter of correct answer (one of A, B, C, or D)]
</output_format>

## Output

Begin by thoughtfully analyzing the provided context within `<document_analysis>` tags. Then present the resulting formatted question answer pair clearly within `<output_format>` tags.

## Important Notes

- Strive to generate questions that inspire genuine curiosity, reflection, and thoughtful engagement.
- Maintain clear, direct, and accurate citations/explanations drawn verbatim from the provided context.
- Ensure complexity and depth reflect thoughtful moderation as guided by the additional instructions.
- Each "thought_process" should reflect careful consideration and reasoning behind your question generation.
- When generating questions, NEVER include phrases like 'as per the text,' 'according to the document,' or any similar explicit references. Questions should inherently integrate content naturally and stand independently without explicit references to the source material. Make sure that the question is answerable by a domain expert **without the context paragraph**. 
- Include all relevant context information in the question. Make the question as long and detailed as required so that the test taker can fully understand what is being asked.
- Do not include answer information in the question. 
- Ensure rigorous adherence to output formatting and generate a single `<output_format>` tag block.
- Ensure that all four answer options are distinct. 
- Verify that the answer options are unambiguous. 
- Verify the correct answer is present. 
- Verify that the question and answer are semantically equivalent to the original question and answer.



<context>{context}</context>
<question>{question}</question>
<answer>{answer}</answer>
"""


TOPIC_EXTRACT_PROMPT = """
## Your Role

You are an expert educational content creator specializing in crafting highly detailed evaluations to determine competency of topic domain experts based on the provided textual information. Your goal is to produce meaningful, highly challenging question-answer pairs that encourage reflection, insight, and nuanced understanding, tailored specifically according to provided instructions.

## Input Structure

Your input consists of:

<context>
[The text segment to analyze, understand, and generate questions about.]
</context>

## Primary Objective

Your goal is to generate a list of all relevant topics, facts, or information that should be included in an examination to evaluate how well a professional domain expert understands the `<context>`. 
The topics should encourage a deep engagement with the content, critically reflect on implications, and clearly demonstrate understanding and competency.

## Analysis Phase

Conduct careful analysis within `<document_analysis>` tags, following these steps:

1. **Thoughtful Content Examination**
   - Carefully analyze the given context, identifying central ideas, nuanced themes, and significant relationships within it.

2. **Concept Exploration**
   - Consider implicit assumptions, subtle details, underlying theories, and potential applications of the provided information.

3. **Strategic Complexity Calibration**
   - Thoughtfully rate difficulty (1-10), ensuring easy questions are avoided.

4. **Intentional Question Planning**
   - Plan how the question topcis can invite deeper understanding, meaningful reflection, or critical engagement, ensuring the questions are purposeful.

## Additional Instructions for Handling Irrelevant or Bogus Information

### Identification and Ignoring of Irrelevant Information:

- **Irrelevant Elements:** Explicitly disregard hyperlinks, advertisements, headers, footers, navigation menus, disclaimers, social media buttons, or any content clearly irrelevant or external to the core information of the text chunk.
- **Bogus Information:** Detect and exclude any information that appears nonsensical or disconnected from the primary subject matter.

### Decision Criteria for Question Generation:

- **Meaningful Content Requirement:** Only generate question topics if the provided `<context>` contains meaningful, coherent, and educationally valuable content.
- **Complete Irrelevance:** If the entire `<context>` consists exclusively of irrelevant, promotional, web navigation, footer, header, or non-informational text, explicitly state this in your analysis and do NOT produce any question-answer pairs.

### Documentation in Analysis:

- Clearly document the rationale in the `<document_analysis>` tags when identifying irrelevant or bogus content, explaining your reasons for exclusion or inclusion decisions.
- Briefly justify any decision NOT to generate questions due to irrelevance or poor quality content.


## Question Topic Generation Guidelines

### Encouraged Question Topic Characteristics:

- **Thoughtful Engagement**: Prioritize creating questions that inspire deeper thought and nuanced consideration.
- **High Complexity**: Develop questions that challenge the domain expert, following the provided additional instructions.
- **Deep Understanding and Insight**: Ensure that the question topics require a deep understanding of the content by a professional domain expert.
- **Self-contained Clarity**: Questions and answers should contain sufficient context, clearly understandable independently of external references.
- **Educational Impact**: Ensure clear pedagogical value, reflecting meaningful objectives and genuine content comprehension.
- **Full Coverage**: Ensure the generated topics fully cover the set of possible evaluation topics for the context for which high quality questions can be generated.

## Output Structure

Present your final output strictly adhering the `<output_format>` tags.
<output_format>
Topic: [ Topic Text ]
Topic: [ Topic Text ]
...
</output_format>

## Output

Begin by thoughtfully analyzing the provided context within `<document_analysis>` tags. Then present the resulting formatted question topics clearly within `<output_format>` tags.

## Important Notes

- Strive to generate question topics that inspire genuine curiosity, reflection, and thoughtful engagement.
- Maintain clear, direct, and accurate citations/explanations drawn verbatim from the provided context.
- Ensure complexity and depth reflect thoughtful moderation as guided by the additional instructions.
- Each "thought_process" should reflect careful consideration and reasoning behind your question topic generation.
- Ensure rigorous adherence to output formatting.
- Get as detailed as required to fully cover the information present in the context.
- The topic response should be a single sentence that fully describes the topic.


<context>{context}</context>
"""




# citation https://github.com/sumukshashidhar/yourbench
# modified
QUESTION_GEN_PROMPT = """
## Your Role

You are an expert educational content creator specializing in crafting highly detailed evaluations to determine competency of topic domain experts based on the provided textual information. Your goal is to produce meaningful, highly challenging question-answer pairs that encourage reflection, insight, and nuanced understanding, tailored specifically according to provided instructions.

## Input Structure

Your input consists of:

<question_topic>
[A topic around which the question should be generated.]
</question_topic>

<context>
[The text segment to analyze, understand, and generate questions about.]
</context>

## Primary Objective

Your goal is to generate a single highly insightful and probing question-answer pair from the single provided `<context>`. Aim for highly technical understanding to probe domain expert knowledge about the `<context>`. The question needs to encourage a deep engagement with the content, critically reflect on implications, and clearly demonstrate understanding and competency. Constructed questions must be highly challenging to even the smartest domain experts.

## Analysis Phase

Conduct careful analysis within `<document_analysis>` tags, following these steps:

1. **Thoughtful Content Examination**
   - Carefully analyze the given context, identifying central ideas, nuanced themes, and significant relationships within it.

2. **Concept Exploration**
   - Consider implicit assumptions, subtle details, underlying theories, and potential applications of the provided information.

3. **Strategic Complexity Calibration**
   - Thoughtfully rate difficulty (1-10), ensuring easy questions are avoided.

4. **Intentional Question Planning**
   - Plan how the question can invite deeper understanding, meaningful reflection, or critical engagement, ensuring the question is purposeful.

## Additional Instructions for Handling Irrelevant or Bogus Information

### Identification and Ignoring of Irrelevant Information:

- **Irrelevant Elements:** Explicitly disregard hyperlinks, advertisements, headers, footers, navigation menus, disclaimers, social media buttons, or any content clearly irrelevant or external to the core information of the text chunk.
- **Bogus Information:** Detect and exclude any information that appears nonsensical or disconnected from the primary subject matter.

### Decision Criteria for Question Generation:

- **Meaningful Content Requirement:** Only generate questions if the provided `<context>` contains meaningful, coherent, and educationally valuable content.
- **Complete Irrelevance:** If the entire `<context>` consists exclusively of irrelevant, promotional, web navigation, footer, header, or non-informational text, explicitly state this in your analysis and do NOT produce any question-answer pairs.

### Documentation in Analysis:

- Clearly document the rationale in the `<document_analysis>` tags when identifying irrelevant or bogus content, explaining your reasons for exclusion or inclusion decisions.
- Briefly justify any decision NOT to generate questions due to irrelevance or poor quality content.


## Question Generation Guidelines

### Encouraged Question Characteristics:

- **Thoughtful Engagement**: Prioritize creating questions that inspire deeper thought and nuanced consideration.
- **High Complexity**: Develop questions that challenge the domain expert, following the provided additional instructions.
- **High Difficulty**: Ensure that the question is very difficult to answer correctly, even for the smartest domain experts.
- **Generalizable**: The best questions require the synthesis of high level general understanding above and beyond the specific context.
- **Deep Understanding and Insight**: Ensure that the question and answers require a deep understanding of the content by a professional domain expert.
- **Self-contained Clarity**: Questions and answers should contain sufficient context, clearly understandable independently of external references.
- **Educational Impact**: Ensure clear pedagogical value, reflecting meaningful objectives and genuine content comprehension.
- **Conversational Tone**: Formulate engaging, natural, and realistic questions appropriate to the instructional guidelines.
- **Short and Factual**: Ensure that the question and answer are short and factual, and that the answer is a single phrase or sentence.

### Permitted Question Types:

- Analytical
- Application-based
- Clarification
- Counterfactual
- Understanding
- Conceptual
- Factual
- Open-ended
- False-premise
- Edge-case
- Inference
- Implication
- Prediction

(You do not need to use every question type, only those naturally fitting the content and instructions.)

## Output Structure

Present your final output strictly adhering the `<output_format>` tags.
<output_format>
Question: [ Question Text ]
A: [ Answer Option A ]
B: [ Answer Option B ]
C: [ Answer Option C ]
D: [ Answer Option D ]
Explanation: [Brief explanation of why the answer is correct]
Correct Answer: [Letter of correct answer (one of A, B, C, or D)]
</output_format>

## Output

Begin by thoughtfully analyzing the provided context within `<document_analysis>` tags. Then present the resulting formatted question answer pair clearly within `<output_format>` tags.

## Important Notes

- Strive to generate a question that inspires genuine curiosity, reflection, and thoughtful engagement.
- Maintain clear, direct, and accurate citations/explanations drawn verbatim from the provided context.
- Each "thought_process" should reflect careful consideration and reasoning behind your question generation and why the question is challenging.
- When generating questions, NEVER include phrases like 'as per the text,' 'according to the document,' or any similar explicit references. Questions should inherently integrate content naturally and stand independently without explicit references to the source material. Make sure that the question is answerable by a domain expert **without the context paragraph**. 
- NEVER include information in the question that could give away the answer. 
- NEVER ask questions where the answer is obvious or apparent.
- Ensure all answer options are plausible and reasonable.
- Ensure at least one incorrect answer is wrong in a subtle way that is very difficult to distinguish from the correct answer.
- Ensure that all four answer options are distinct. 
- Verify the correct answer is present. 
- Ensure rigorous adherence to output formatting and generate a single `<output_format>` tag block.


<question_topic>{topic}</question_topic>
<context>{context}</context>
"""
# - Ensure complexity and depth reflect thoughtful moderation as guided by the additional instructions.






# citation https://github.com/sumukshashidhar/yourbench
# modified
QUESTION_GEN_OPEN_PROMPT = """
## Your Role

You are an expert educational content creator specializing in crafting highly detailed evaluations to determine competency of topic domain experts based on the provided textual information. Your goal is to produce meaningful, highly challenging question-answer pairs that encourage reflection, insight, and nuanced understanding, tailored specifically according to provided instructions.

## Input Structure

Your input consists of:

<context>
[The text segment to analyze, understand, and generate questions about.]
</context>

<question_topic>
[A topic around which the question should be generated.]
</question_topic>

## Primary Objective

Your goal is to generate a single highly insightful and probing question-answer pair from the single provided `<context>`. Aim for highly technical understanding to probe domain expert knowledge about the `<context>`. The question needs to encourage a deep engagement with the content, critically reflect on implications, and clearly demonstrate understanding and competency. Constructed questions must be highly challenging to even the smartest domain experts.

## Analysis Phase

Conduct careful analysis within `<document_analysis>` tags, following these steps:

1. **Thoughtful Content Examination**
   - Carefully analyze the given context, identifying central ideas, nuanced themes, and significant relationships within it.

2. **Concept Exploration**
   - Consider implicit assumptions, subtle details, underlying theories, and potential applications of the provided information.

3. **Strategic Complexity Calibration**
   - Thoughtfully rate difficulty (1-10), ensuring easy questions are avoided.

4. **Intentional Question Planning**
   - Plan how the question can invite deeper understanding, meaningful reflection, or critical engagement, ensuring the question is purposeful.

## Additional Instructions for Handling Irrelevant or Bogus Information

### Identification and Ignoring of Irrelevant Information:

- **Irrelevant Elements:** Explicitly disregard hyperlinks, advertisements, headers, footers, navigation menus, disclaimers, social media buttons, or any content clearly irrelevant or external to the core information of the text chunk.
- **Bogus Information:** Detect and exclude any information that appears nonsensical or disconnected from the primary subject matter.

### Decision Criteria for Question Generation:

- **Meaningful Content Requirement:** Only generate questions if the provided `<context>` contains meaningful, coherent, and educationally valuable content.
- **Complete Irrelevance:** If the entire `<context>` consists exclusively of irrelevant, promotional, web navigation, footer, header, or non-informational text, explicitly state this in your analysis and do NOT produce any question-answer pairs.

### Documentation in Analysis:

- Clearly document the rationale in the `<document_analysis>` tags when identifying irrelevant or bogus content, explaining your reasons for exclusion or inclusion decisions.
- Briefly justify any decision NOT to generate questions due to irrelevance or poor quality content.


## Question Generation Guidelines

### Encouraged Question Characteristics:

- **Thoughtful Engagement**: Prioritize creating questions that inspire deeper thought and nuanced consideration.
- **High Complexity**: Develop questions that challenge the domain expert, following the provided additional instructions.
- **High Difficulty**: Ensure that the question is very difficult to answer correctly, even for the smartest domain experts.
- **Generalizable**: The best questions require the synthesis of high level general understanding above and beyond the specific context.
- **Deep Understanding and Insight**: Ensure that the question and answers require a deep understanding of the content by a professional domain expert.
- **Self-contained Clarity**: Questions and answers should contain sufficient context, clearly understandable independently of external references.
- **Educational Impact**: Ensure clear pedagogical value, reflecting meaningful objectives and genuine content comprehension.
- **Conversational Tone**: Formulate engaging, natural, and realistic questions appropriate to the instructional guidelines.
- **Short and Factual**: Ensure that the question and answer are short and factual, and that the answer is a single phrase or sentence.

### Permitted Question Types:

- Analytical
- Application-based
- Clarification
- Counterfactual
- Understanding
- Conceptual
- Factual
- Open-ended
- False-premise
- Edge-case
- Inference
- Implication
- Prediction

(You do not need to use every question type, only those naturally fitting the content and instructions.)

## Output Structure

Present your final output strictly adhering the `<output_format>` tags.
<output_format>
Question: [ Question Text ]
Explanation: [Brief explanation of why the answer is correct]
Correct Answer: [Short answer]
</output_format>

## Output

Begin by thoughtfully analyzing the provided context within `<document_analysis>` tags. Then present the resulting formatted question answer pair clearly within `<output_format>` tags.

## Important Notes

- Strive to generate a question that inspires genuine curiosity, reflection, and thoughtful engagement.
- Maintain clear, direct, and accurate citations/explanations drawn verbatim from the provided context.
- Each "thought_process" should reflect careful consideration and reasoning behind your question generation.
- When generating questions, NEVER include phrases like 'as per the text,' 'according to the document,' or any similar explicit references. Questions should inherently integrate content naturally and stand independently without explicit references to the source material. Make sure that the question is answerable by a domain expert **without the context paragraph**. 
- NEVER include information in the question that could give away the answer.
- NEVER ask questions where the answer is obvious or apparent.
- Verify that the correct answer is in fact correct and the best version of that answer.
- Ensure rigorous adherence to output formatting and generate a single `<output_format>` tag block.


<context>{context}</context>
<question_topic>{topic}</question_topic>
"""
# - Ensure complexity and depth reflect thoughtful moderation as guided by the additional instructions.


QUESTION_REFORMAT_OPEN_PROMPT = """
## Your Role

You are an expert educational content creator specializing in crafting highly detailed evaluations to determine competency of topic domain experts based on the provided textual information. Your goal is to produce meaningful, highly challenging question-answer pairs that encourage reflection, insight, and nuanced understanding, tailored specifically according to provided instructions.

## Input Structure

Your input consists of:

<context>
[The text segment containing information relevant to the question.]
</context>

<question>
[A question to be answered.]
</question>

<answer>
[The correct answer to the question.]
</answer>

## Primary Objective

Your goal is to reformat, rephrase, and rewrite the question and answer pair according to the provided instructions. The rewritten question should be semantically equivalent to the original question, rewritten for clarity. The rewritten answer should be semantically equivalent to the original answer.

## Analysis Phase

Conduct careful analysis within `<document_analysis>` tags, following these steps:

1. **Thoughtful Content Examination**
   - Carefully analyze the given context, question, and answer;identifying central ideas, nuanced themes, and significant relationships within it.

2. **Concept Exploration**
   - Consider implicit assumptions, subtle details, underlying theories, and potential applications of the provided information.

3. **Strategic Complexity Calibration**
   - Thoughtfully rate difficulty (1-10), ensuring easy questions are avoided.

4. **Intentional Question Planning**
   - Plan how the question can invite deeper understanding, meaningful reflection, or critical engagement, ensuring the question is purposeful.

### Documentation in Analysis:

- Clearly document the rationale in the `<document_analysis>` tags when identifying irrelevant or bogus content, explaining your reasons for exclusion or inclusion decisions.
- Briefly justify any decision NOT to generate questions due to irrelevance or poor quality content.


## Question Generation Guidelines

### Encouraged Question Characteristics:

- **Thoughtful Engagement**: Prioritize creating questions that inspire deeper thought and nuanced consideration.
- **High Complexity**: Develop questions that challenge the domain expert, following the provided additional instructions.
- **Deep Understanding and Insight**: Ensure that the question and answers require a deep understanding of the content by a professional domain expert.
- **Self-contained Clarity**: Questions and answers should contain sufficient context, clearly understandable independently of external references.
- **Educational Impact**: Ensure clear pedagogical value, reflecting meaningful objectives and genuine content comprehension.
- **Conversational Tone**: Formulate engaging, natural, and realistic questions appropriate to the instructional guidelines.

### Permitted Question Types:

- Analytical
- Application-based
- Clarification
- Counterfactual
- Conceptual
- Factual
- Open-ended
- False-premise
- Edge-case

(You do not need to use every question type, only those naturally fitting the content and instructions.)

## Output Structure

Present your final output strictly adhering the `<output_format>` tags.
<output_format>
Question: [ Question Text ]
Explanation: [Brief explanation of why the answer is correct]
Correct Answer: [Short answer]
</output_format>

## Output

Begin by thoughtfully analyzing the provided context within `<document_analysis>` tags. Then present the resulting formatted question answer pair clearly within `<output_format>` tags.

## Important Notes

- Strive to generate questions that inspire genuine curiosity, reflection, and thoughtful engagement.
- Maintain clear, direct, and accurate citations/explanations drawn verbatim from the provided context.
- Ensure complexity and depth reflect thoughtful moderation as guided by the additional instructions.
- Each "thought_process" should reflect careful consideration and reasoning behind your question generation.
- When generating questions, NEVER include phrases like 'as per the text,' 'according to the document,' or any similar explicit references. Questions should inherently integrate content naturally and stand independently without explicit references to the source material. Make sure that the question is answerable by a domain expert **without the context paragraph**. 
- Include all relevant context information in the question. Make the question as long and detailed as required so that the test taker can fully understand what is being asked.
- Do not include answer information in the question. 
- Ensure rigorous adherence to output formatting and generate a single `<output_format>` tag block.
- Verify that the correct answer is in fact corrent and the best version of that answer.
- Verify that the question and answer are semantically equivalent to the original question and answer.



<context>{context}</context>
<question>{question}</question>
<answer>{answer}</answer>
"""