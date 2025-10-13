"""This file contains prompts for counterfactual analysis in the biography generation task."""

################################################################################
#    Prompt for flipping the factual correctness in the first sentence         #
################################################################################  
_ORIGINAL_FIRST_SENT = '[FIRST_SENT]'
_ALL_SUPPORTED_FACTS = '[SUPPORTED_FACTS]'

FLIP_FACTUALITY_PROMPT = f"""
You are a helpful assistant. You will be given a one-sentence bio of an entity. \
There are supported and unsupported facts in the bio. You need to convert one supported fact into an unsupported fact \
or generate new unsupported facts. \
And then you should give a new bio including the new unsupported facts. \
The new bio should keep the syntax and structure of the original bio while introducing a small factual error. The new bio should still be one sentence.

Here are the guidelines for generating new unsupported facts:
1. Keep it plausible: The new unsupported facts should NOT alter the main point of the original bio. It should introduce small perturbations rather than major shifts in context.
2. The overall meaning should NOT change dramatically. **Small factual errors (e.g., places, dates or minor career details) are suitable**.
3. You can generate unsupported facts by slightly altering the supported facts, referring to the original unsupported facts, \
or generating plausible but unsupported details, or other ways.
4. Keep the provided unsupported facts in the new bio.
5. The inserted unsupported fact should relate to the broader biography and fit into the narrative.

You need to firstly give new unsupported facts. Then you need to give a new bio including the new unsupported facts. The new bio should match the format of the original bio as closely as possible.

The response format should be:
New unsupported facts: [new unsupported facts]
New bio: [new bio]
""".strip()

flip_factuality_instruction = f"""
- Orignial bio: {_ORIGINAL_FIRST_SENT}
- Supported facts: {_ALL_SUPPORTED_FACTS}
""".strip()

################################################################################
#          Prompt for continuing generation from the first sentence            #
################################################################################  
_TOPIC_PLACEHOLDER = '[TOPIC]'
_FIRST_SENTENCE_BIO = '[FIRST_SENTENCE_BIO]'

CONTINUE_GEN_PROMPT = f"""
You are a helpful assistant. You will be given an entity name and the first sentence in the bio for it. \
You need to complete the given bio. Here are the instructions: \
1. Be sure to only include accurate, factual information in the completed bio. \
2. The completed bio should be comprehensive and detailed.\
3. Do NOT change the given one-sentence bio. The completed bio should start with the given first sentence bio. \
4. Return ONLY the completed bio, and nothing else.
""".strip()

continue_gen_instruction = f"""
Complete the following bio of {_TOPIC_PLACEHOLDER}.

The first sentence in the bio: {_FIRST_SENTENCE_BIO}
""".strip()
