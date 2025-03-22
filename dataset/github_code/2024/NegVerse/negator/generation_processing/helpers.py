

#from ..PEFT.data_preprocess import Special_tokens

import re

class Special_tokens():
    PERETURB_TOK = "<|perturb|>"
    BLANK_TOK = "[BLANK]"
    SEP_TOK = "[SEP]"
    ANSWER_TOK = "[ANSWER]"
    NEG_TOK = "[negation]"
    EMPTY_TOK = "[EMPTY]"

def remove_blanks(total_sequence: str ) -> str:

    """
    Fills in the blanks in a template sentence with provided answers from the total sequence.

    The input string should have a special token '[NEG]' to mark the start of the relevant part,
    a '[SEP]' token to separate the template and answers, and each answer is marked by '[ANSWER]'.
    An '[EMPTY]' token is used to indicate an empty answer.

    Parameters:
    total_sequence (str): The input string containing the sequence with the template sentence and answers.

    Returns:
    tuple: A tuple containing the completed sentence and the list of answers.
    """
        
    try:
        text = total_sequence.split(Special_tokens.NEG_TOK)[-1]
        before, answers = text.split(Special_tokens.SEP_TOK)
        answers = [x.strip() for x in answers.split(Special_tokens.ANSWER_TOK)][:-1]
        answers = [x if x != Special_tokens.EMPTY_TOK else '' for x in answers]
        for a in answers:
            if a == '':
                before = re.sub(r' %s' % re.escape(Special_tokens.BLANK_TOK), a, before, count=1)
            else:
                before = re.sub(r'%s' % re.escape(Special_tokens.BLANK_TOK), a, before, count=1)
            final = before.split('!')[0] #remove pad token
        return final, answers
    except:
        return text, []
    


def get_prompts(doc, blanked_sents, is_complete_blank=True):
    prompts = []
    tag = 'negation'
    for bt in blanked_sents:
        sep_tok = Special_tokens.SEP_TOK if bt and is_complete_blank else ""
        prompts.append(f"{doc.text.strip()} {Special_tokens.PERETURB_TOK} [{tag}] {bt.strip()} {sep_tok}".strip())
    return prompts


def get_outputs(model,inputs,do_sample=True,num_beams=None,num_return_sequences = 3):
    """
    Generates multiple sequences of text using the provided model and inputs.

    Args:
        model: The model used for generation.
        inputs (dict): Input tensors including 'input_ids' and 'attention_mask'.
        do_sample (bool, optional): Whether to use sampling during generation (default: True).
        num_beams (int, optional): Number of beams for beam search. Overrides `do_sample`.
        num_return_sequences (int, optional): Number of sequences to generate per input (default: 3).

    Returns:
        torch.Tensor: Tensor containing generated sequences.
    """

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=1000,
        early_stopping=False, #if num_beams is None else True, #The model can stop before reach the max_length
        temperature= 1,
        num_beams=1 if num_beams is None else num_beams,
        do_sample=num_beams is None and do_sample,
        num_return_sequences=num_return_sequences,
        pad_token_id= 50256 #tokenizer.eos_token_id
    )
    return outputs


