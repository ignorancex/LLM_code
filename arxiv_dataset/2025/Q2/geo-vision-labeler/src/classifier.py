# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


def classify_with_openai(
    description, include_filename, filename, classes, openai_client, model_name="gpt-4"
):
    """
    Uses the OpenAI API to classify the image description into one of the classes.
    The prompt instructs the model to output exactly one of the classes.

    Args:
        description (str): The description of the image.
        filename (str): The filename of the image.
        classes (list): The list of classes to classify into.
        openai_client: The OpenAI API client instance.
        model_name (str): The name of the OpenAI model to use.
    Returns:
        str: The classification result.
    """

    if include_filename:
        filename_str = f"Use the filename [{filename}] to locate the place and use that in your prediction. "
    else:
        filename_str = ""

    messages = [
        {"role": "system", "content": "You are a classifier."},
        {
            "role": "user",
            "content": (
                f"Given a text describing a place, output one of the following classes: {', '.join(classes)}. "
                f"{filename_str}"
                f"Don't output anything else but one of the classes: {description}. Description: {description}. Class: "
            ),
        },
    ]
    response = openai_client.chat.completions.create(
        model=model_name, messages=messages
    )
    classification = response.choices[0].message.content.strip()
    return classification


def classify_with_huggingface(
    description,
    include_filename,
    filename,
    classes,
    pipeline_instance,
    tokenizer,
    max_new_tokens=10,
):
    """
    Uses an open-source language model via a Hugging Face text-generation pipeline to classify the image description.

    Args:
        description (str): The description of the image.
        filename (str): The filename of the image.
        classes (list): The list of classes to classify into.
        pipeline_instance: The Hugging Face pipeline instance.
        tokenizer: The tokenizer for the model.
        max_new_tokens (int): The maximum number of tokens to generate.
    Returns:
        str: The classification result.
    """

    if include_filename:
        filename_str = f"Use the filename [{filename}] to locate the place and use that in your prediction. "
    else:
        filename_str = ""

    messages = [
        {"role": "system", "content": "You are a classifier."},
        {
            "role": "user",
            "content": (
                f"Given a text describing a place, output one of the following classes: {', '.join(classes)}. "
                f"{filename_str}"
                f"Don't output anything else but one of the classes. Description: {description}. Class: "
            ),
        },
    ]
    response = pipeline_instance(
        messages,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=1.0,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    classification = response[0]["generated_text"]

    if classification[-1] in [".", ",", "!", "?"]:
        classification = classification[:-1]
    classification = classification.strip()

    return classification
