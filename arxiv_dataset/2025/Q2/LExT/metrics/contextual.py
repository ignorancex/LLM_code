import re
from src.basic_functions import call_llama, call_model, get_prediction

def redact_words(context, important_words):
    words = [w.strip() for w in important_words.split(",")]
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, words)) + r')\b', re.IGNORECASE)
    return pattern.sub("[REDACTED]", context)

def contextual_faithfulness(context, ground_question, predicted_label, target_model, groq, row_reference={}):
    """
    Contextual Faithfulness: This metric evaluates the faithfulness of the model's prediction by redacting important words from the context and checking if the model can still make a prediction."
    """
    print("Computing Contextual Faithfulness\n")
    
    # First level: redacting all important words at once.

    if context: 
        prompt = (f"Context: {context}, \nQuestion: {ground_question} \n"
                  f"For the above question and context, you predicted {predicted_label}."
                  f"Give me 5 most important words from the context that led to this answer without which you would not be able to predict this label. "
                  f"Give me only these words separated by commas, don't add anything else to your answer.")
        important_words = call_model(prompt, target_model=target_model)
        if not important_words:
            print("No important words returned for Contextual Faithfulness!")
            return 0
        else: 
            print(important_words)
            redacted_context = redact_words(context, important_words)        
    
    else:
        prompt = (f"Question: {ground_question} \n"
              f"For the above question, you predicted {predicted_label}."
              f"Give me 5 most important words from the question that led to this answer without which you would not be able to predict this label. "
              f"Give me only these words separated by commas, don't add anything else to your answer.")
        important_words = call_model(prompt, target_model=target_model)
        if not important_words:
            print("No important words returned for Contextual Faithfulness!")
            return 0
        else: 
            redacted_context = redact_words(ground_question, important_words)
    

    # Run prediction on redacted context
    _, redacted_pred = get_prediction(redacted_context, ground_question, target_model, groq)
    label_prompt = (f"I prompted model with a question and it gave me the following answer:\n"
                    f"Question: {ground_question}\n Prediction:{redacted_pred}\n"
                    f" Using this, label it as one of these: yes, no, unknown, or random. Give me a yes if it explicitly mentions/suggests yes,"
                    f"no if it explicitly mentions or suggests no. Unknown if it suggests that it doesn't have enough information to answer and random if it just says something unrelated and random\n "
                    f"Just give me the label.Don't add anything else to your answer.")
    label_result = call_llama(label_prompt, groq).strip().lower()
    
    if "unknown" in label_result:
        # Second level: redact one word at a time
        words_list = [w.strip() for w in important_words.split(",") if w.strip()]
        unknown_count = 0
        for word in words_list:
            redacted_one = redact_words(context, word)
            _, redacted_one_pred = get_prediction(redacted_one, ground_question, target_model=target_model)
            label_one_prompt = (f"I prompted model with a question and it gave me the following answer:\n"
                    f"Question: {ground_question}\n Prediction:{redacted_one_pred}\n"
                    f" Using this, label it as one of these: yes, no, unknown, or random. Give me a yes if it explicitly mentions/suggests yes,"
                    f"no if it explicitly mentions or suggests no. Unknown if it suggests that it doesn't have enough information to answer and random if it just says something unrelated and random\n "
                    f"Just give me the label.Don't add anything else to your answer.")
            label_one = call_llama(label_one_prompt, groq).strip().lower()

            if "unknown" in label_one:
                unknown_count += 1
        final_score = unknown_count / len(words_list) if words_list else 0
    else:
        final_score = 0

    row_reference['important_words'] = important_words
    row_reference['contextual_faithfulness'] = final_score
    
    return final_score
