import replicate

def model_query(neg_word, pos_word, model_params=None):
    """
    Generate sentences with given negative and positive words using a language model (LLAMA).

    Args:
        neg_word (str): The negative word to use in the sentences.
        pos_word (str): The positive word to use in the sentences.
        model_params (dict, optional): A dictionary of parameters for the model. Defaults to None.

    Returns:
        generator object: The response from the language model.
    """
    # Define the pre-prompt and prompt input with placeholders for the words
    pre_prompt = "You are a helpful assistant. You generate neutral, informative, and contextually appropriate sentences."
    prompt_input = f"Create two neutral and contextually appropriate sentences using the word '{neg_word}'. Then, provide the same sentences but with the word '{pos_word}', keeping minimal changes."

    # Default model parameters if not provided
    default_params = {
        "temperature": 0.1,
        "top_p": 0.9,
        "max_length": 500,
        "repetition_penalty": 1
    }

    # Update default parameters with any provided parameters
    if model_params is not None:
        default_params.update(model_params)

    # Generate LLM response
    output = replicate.run(
        'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5',  # LLM model
        input={
            "prompt": f"{pre_prompt} {prompt_input} Assistant: ",  # Prompts
            **default_params  # Unpack the parameters dictionary
        }
    )
    
    return output


