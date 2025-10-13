import numpy as np
import re


def parse_topic_extraction(response: str) -> dict:
   
    result = list()
    
    # Preprocess the response to remove markdown formatting
    # Replace **text** with text to handle bold formatting
    cleaned_response = re.sub(r'\*\*([^*]+)\*\*', r'\1', response)
    # Remove dash prefixes at the beginning of lines
    cleaned_response = re.sub(r'(?:^|\n)\s*-\s*', r'\n', cleaned_response)
    # Remove leading whitespace from all lines
    cleaned_response = re.sub(r'(?:^|\n)\s+', r'\n', cleaned_response)

    # Extract content between <output_format> and </output_format> tags if present
    output_format_pattern = r'<output_format>(.*?)</output_format>'
    output_format_matches = list(re.finditer(output_format_pattern, cleaned_response, re.DOTALL))
    
    if output_format_matches:
        # If output format tags are found, use only the content from the last match
        cleaned_response = output_format_matches[-1].group(1).strip()

    # Find all lines starting with Topic: and extract the rest of the line
    topic_pattern = r'(?:^|\n)\s*Topic\s*:\s*([^\n]+)'
    topics = re.findall(topic_pattern, cleaned_response)
    
    # Add each found topic to the result list
    for topic in topics:
        result.append(topic.strip())

    return result


def parse_explanation_validity_numbers(response: str, valid_options: list = None) -> dict:
   
    result = {
        'answer_correctness_score': None,
        'explanation_validity_score': None
    }
    
    # Preprocess the response to remove markdown formatting
    # Replace **text** with text to handle bold formatting
    cleaned_response = re.sub(r'\*\*([^*]+)\*\*', r'\1', response)
    # Remove dash prefixes at the beginning of lines
    cleaned_response = re.sub(r'(?:^|\n)\s*-\s*', r'\n', cleaned_response)
    # Remove leading whitespace from all lines
    cleaned_response = re.sub(r'(?:^|\n)\s+', r'\n', cleaned_response)

    # Extract content between <output_format> and </output_format> tags if present
    output_format_pattern = r'<output_format>(.*?)</output_format>'
    output_format_matches = list(re.finditer(output_format_pattern, cleaned_response, re.DOTALL))
    
    if output_format_matches:
        # If output format tags are found, use only the content from the last match
        cleaned_response = output_format_matches[-1].group(1).strip()


    
    answer_match = re.search(r'(?:Answer Correctness|answer correctness)\s*:\s*(\d+)', cleaned_response)
    if answer_match:
        try:
            num = int(answer_match.group(1))
            # If valid_options is provided, check if this number is valid
            if valid_options is None or num in valid_options:
                result['answer_correctness_score'] = num
        except ValueError:
            pass

    answer_match = re.search(r'(?:Explanation Validity|explanation validity)\s*:\s*(\d+)', cleaned_response)
    if answer_match:
        try:
            num = int(answer_match.group(1))
            # If valid_options is provided, check if this number is valid
            if valid_options is None or num in valid_options:
                result['explanation_validity_score'] = num
        except ValueError:
            pass

    for k in result.keys():
        if result[k] is None:
            return None

    return result


def parse_reformat_validity_numbers(response: str, valid_options: list = None) -> dict:
   
    result = {
        'question_similarity_score': None,
        'answer_similarity_score': None
    }
    
    # Preprocess the response to remove markdown formatting
    # Replace **text** with text to handle bold formatting
    cleaned_response = re.sub(r'\*\*([^*]+)\*\*', r'\1', response)
    # Remove dash prefixes at the beginning of lines
    cleaned_response = re.sub(r'(?:^|\n)\s*-\s*', r'\n', cleaned_response)
    # Remove leading whitespace from all lines
    cleaned_response = re.sub(r'(?:^|\n)\s+', r'\n', cleaned_response)

    # Extract content between <output_format> and </output_format> tags if present
    output_format_pattern = r'<output_format>(.*?)</output_format>'
    output_format_matches = list(re.finditer(output_format_pattern, cleaned_response, re.DOTALL))
    
    if output_format_matches:
        # If output format tags are found, use only the content from the last match
        cleaned_response = output_format_matches[-1].group(1).strip()


    
    answer_match = re.search(r'(?:Question Similarity|question similarity)\s*:\s*(\d+)', cleaned_response)
    if answer_match:
        try:
            num = int(answer_match.group(1))
            # If valid_options is provided, check if this number is valid
            if valid_options is None or num in valid_options:
                result['question_similarity_score'] = num
        except ValueError:
            pass

    answer_match = re.search(r'(?:Answer Similarity|answer similarity)\s*:\s*(\d+)', cleaned_response)
    if answer_match:
        try:
            num = int(answer_match.group(1))
            # If valid_options is provided, check if this number is valid
            if valid_options is None or num in valid_options:
                result['answer_similarity_score'] = num
        except ValueError:
            pass

    for k in result.keys():
        if result[k] is None:
            return None

    return result



def parse_meta_properties_numbers(response: str, valid_options: list = None) -> dict:
   
    result = {
        'clarity_score': None,
        'difficulty_score': None,
        'groundedness_score': None
    }
    
    # Preprocess the response to remove markdown formatting
    # Replace **text** with text to handle bold formatting
    cleaned_response = re.sub(r'\*\*([^*]+)\*\*', r'\1', response)
    # Remove dash prefixes at the beginning of lines
    cleaned_response = re.sub(r'(?:^|\n)\s*-\s*', r'\n', cleaned_response)
    # Remove leading whitespace from all lines
    cleaned_response = re.sub(r'(?:^|\n)\s+', r'\n', cleaned_response)

    # Extract content between <output_format> and </output_format> tags if present
    output_format_pattern = r'<output_format>(.*?)</output_format>'
    output_format_matches = list(re.finditer(output_format_pattern, cleaned_response, re.DOTALL))
    
    if output_format_matches:
        # If output format tags are found, use only the content from the last match
        cleaned_response = output_format_matches[-1].group(1).strip()


    
    answer_match = re.search(r'(?:Clarity|clarity)\s*:\s*(\d+)', cleaned_response)
    if answer_match:
        try:
            num = int(answer_match.group(1))
            # If valid_options is provided, check if this number is valid
            if valid_options is None or num in valid_options:
                result['clarity_score'] = num
        except ValueError:
            pass

    answer_match = re.search(r'(?:Difficulty|difficulty)\s*:\s*(\d+)', cleaned_response)
    if answer_match:
        try:
            num = int(answer_match.group(1))
            # If valid_options is provided, check if this number is valid
            if valid_options is None or num in valid_options:
                result['difficulty_score'] = num
        except ValueError:
            pass

    answer_match = re.search(r'(?:Groundedness|groundedness)\s*:\s*(\d+)', cleaned_response)
    if answer_match:    
        try:
            num = int(answer_match.group(1))
            # If valid_options is provided, check if this number is valid
            if valid_options is None or num in valid_options:
                result['groundedness_score'] = num
        except ValueError:
            pass

    for k in result.keys():
        if result[k] is None:
            return None

    return result



def parse_generated_mcq(response: str) -> dict:
    """
    Parses an LLM response to extract a question, correct answer, and answer options.
    
    Args:
        response (str): The full response text from the LLM
        
    Returns:
        dict: A dictionary containing:
            - 'question': The extracted question string
            - 'correct_answer': The correct answer string
            - 'options': A dictionary of answer options (e.g., {'A': 'option text', 'B': 'option text'})
            - 'response': The original response
            
    Raises:
        ValueError: If required components cannot be extracted
    """
    result = {
        'question': None,
        'correct_answer': None,
        'options': {},
        'response': response,
        'explanation': None
    }
    
    # Preprocess the response to remove markdown formatting
    # Replace **text** with text to handle bold formatting
    cleaned_response = re.sub(r'\*\*([^*]+)\*\*', r'\1', response)
    # Remove dash prefixes at the beginning of lines
    cleaned_response = re.sub(r'(?:^|\n)\s*-\s*', r'\n', cleaned_response)
    # Remove leading whitespace from all lines
    cleaned_response = re.sub(r'(?:^|\n)\s+', r'\n', cleaned_response)
    # Remove markdown headings (# ## and ###)
    cleaned_response = re.sub(r'(?:^|\n)\s*#{1,3}\s+', r'\n', cleaned_response)

    # Extract content between <output_format> and </output_format> tags if present
    output_format_pattern = r'<output_format>(.*?)</output_format>'
    output_format_matches = list(re.finditer(output_format_pattern, cleaned_response, re.DOTALL))
    
    if output_format_matches:
        # If output format tags are found, use only the content from the last match
        cleaned_response = output_format_matches[-1].group(1).strip()
    
    
    # Try to extract the question - now supporting both formats
    question_patterns = [
        # Format: Question: text (single line only)
        r"(?:^|\n)\s*Question\s*:?\s*([^\n]+)(?=\s*(?:[A-D]:|Options|Choices|Answers|A\s*[:.)]|A\.|A\)|\(A\)|A:))",
        # Fallback pattern (single line only)
        r"(?:^|\n)\s*Question\s*:?\s*([^\n]+)",
        # Format with newline-separated options (single line only)
        r"(?:^|\n)\s*Question\s*:?\s*([^\n]+)(?=\s*A:)",
    ]
    
    for pattern in question_patterns:
        # Find all matches and take the last one
        question_matches = list(re.finditer(pattern, cleaned_response))
        if question_matches:
            result['question'] = question_matches[-1].group(1).strip()
            break

    # Try to extract the question - now supporting both formats
    explanation_patterns = [
        # Format: Explanation: text (single line only)
        r"(?:^|\n)\s*Explanation\s*:\s*([^\n]+)(?=\s*(?:Correct\s+Answer|$))",
        # Format: Explanation - text
        r"(?:^|\n)\s*Explanation\s*-\s*([^\n]+)",
        # Format: Explanation (with optional colon)
        r"(?:^|\n)\s*Explanation\s*:?\s*([^\n]+)"
    ]
    
    for pattern in explanation_patterns:
        # Find all matches and take the last one
        explanation_matches = list(re.finditer(pattern, cleaned_response))
        if explanation_matches:
            result['explanation'] = explanation_matches[-1].group(1).strip()
            break
    
    # Extract options
    option_patterns = [
        # Format: A: Option text
        r"(?:^|\n)\s*(?!Correct\s+Answer)([A-D])\s*[:.)]?\s*([^\n]+)",
        # Format: (A) Option text
        r"(?:^|\n)\s*(?!Correct\s+Answer)\(([A-D])\)\s*([^\n]+)",
        # New format with newline-separated options
        r"(?:^|\n)(?!Correct\s+Answer)([A-D]):\s*([^\n]+)(?=\n|$)",
    ]
    
    for pattern in option_patterns:
        options = re.findall(pattern, cleaned_response) #, re.MULTILINE)
        if options:
            for option_letter, option_text in options:
                result['options'][option_letter.upper()] = option_text.strip()
            break
    
    # Try to extract the correct answer
    answer_patterns = [
        # Format: Correct Answer: X) with text after
        r"(?:^|\n)\s*Correct\s+Answer\s*:?\s*([A-D])\s*[\)\.](.*)$",
        # Format: Correct Answer: X with text after
        r"(?:^|\n)\s*Correct\s+Answer\s*:?\s*([A-D])(?:\s|$|[^A-D])(.*)$",
        # Format: Correct Answer: (X) with text after
        r"(?:^|\n)\s*Correct\s+Answer\s*:?\s*\(([A-D])\)(.*)$",
        # Format: Correct Answer is/= X with text after
        r"(?:^|\n)\s*Correct\s+Answer\s*(?:is|=)\s*([A-D])(.*)$",
    ]
    
    for pattern in answer_patterns:
        answer_match = re.search(pattern, cleaned_response)
        if answer_match:
            result['correct_answer'] = answer_match.group(1).upper()
            # Store the text after the answer if it exists
            if len(answer_match.groups()) > 1 and answer_match.group(2):
                result['answer_explanation'] = answer_match.group(2).strip()
            break
    
    # If we couldn't find the answer in the cleaned response, try the original response
    # This is a fallback in case the cleaning process affected the answer patterns
    if not result['correct_answer']:
        for pattern in answer_patterns:
            answer_match = re.search(pattern, response) #, re.MULTILINE)
            if answer_match:
                result['correct_answer'] = answer_match.group(1).upper()
                # Store the text after the answer if it exists
                if len(answer_match.groups()) > 1 and answer_match.group(2):
                    result['answer_explanation'] = answer_match.group(2).strip()
                break
    
    # Final validation
    if not result['question'] or not result['options'] or not result['correct_answer'] or not result['explanation']:
        return None

    # Additional validation for correct_answer
    if result['correct_answer'].strip() == "" or result['explanation'].strip() == "" or result['question'].strip() == "":
        return None

    # Validate that options only contain keys A, B, C, D
    valid_keys = {"A", "B", "C", "D"}
    invalid_keys = set(result['options'].keys()) - valid_keys
    missing_keys = valid_keys - set(result['options'].keys())
    
    if invalid_keys:
        # Remove any invalid keys
        for key in invalid_keys:
            del result['options'][key]
    if missing_keys:
        return None
        
    # Check if correct_answer is a valid option
    if result['correct_answer'] not in valid_keys or result['correct_answer'] not in result['options']:
        return None
    
    # Confirm that for every valid key, options has a non-empty string
    for key in result['options'].keys():
        if not result['options'][key] or result['options'][key].strip() == "":
            return None
        
    # Confirm that each option is non-None
    for key, value in result['options'].items():
        if value is None:
            return None
    
    return result



def parse_generated_open(response: str) -> dict:
    """
    Parses an LLM response to extract a question, correct answer, and answer options.
    
    Args:
        response (str): The full response text from the LLM
        
    Returns:
        dict: A dictionary containing:
            - 'question': The extracted question string
            - 'correct_answer': The correct answer string
            - 'response': The original response
            
    Raises:
        ValueError: If required components cannot be extracted
    """
    result = {
        'question': None,
        'correct_answer': None,
        'response': response,
        'explanation': None
    }
    
    # Preprocess the response to remove markdown formatting
    # Replace **text** with text to handle bold formatting
    cleaned_response = re.sub(r'\*\*([^*]+)\*\*', r'\1', response)
    # Remove dash prefixes at the beginning of lines
    cleaned_response = re.sub(r'(?:^|\n)\s*-\s*', r'\n', cleaned_response)
    # Remove leading whitespace from all lines
    cleaned_response = re.sub(r'(?:^|\n)\s+', r'\n', cleaned_response)
    # Remove markdown headings (# ## and ###)
    cleaned_response = re.sub(r'(?:^|\n)\s*#{1,3}\s+', r'\n', cleaned_response)

    # Extract content between <output_format> and </output_format> tags if present
    output_format_pattern = r'<output_format>(.*?)</output_format>'
    output_format_matches = list(re.finditer(output_format_pattern, cleaned_response, re.DOTALL))
    
    if output_format_matches:
        # If output format tags are found, use only the content from the last match
        cleaned_response = output_format_matches[-1].group(1).strip()
    
    
    # Try to extract the question - now supporting both formats
    question_patterns = [
        # Format: Question: text (single line only)
        r"(?:^|\n)\s*Question\s*:?\s*([^\n]+)(?=\s*(?:[A-D]:|Options|Choices|Answers|A\s*[:.)]|A\.|A\)|\(A\)|A:))",
        # Fallback pattern (single line only)
        r"(?:^|\n)\s*Question\s*:?\s*([^\n]+)",
        # Format with newline-separated options (single line only)
        r"(?:^|\n)\s*Question\s*:?\s*([^\n]+)(?=\s*A:)",
    ]
    
    for pattern in question_patterns:
        # Find all matches and take the last one
        question_matches = list(re.finditer(pattern, cleaned_response))
        if question_matches:
            result['question'] = question_matches[-1].group(1).strip()
            break

    # Try to extract the question - now supporting both formats
    explanation_patterns = [
        # Format: Explanation: text (single line only)
        r"(?:^|\n)\s*Explanation\s*:\s*([^\n]+)(?=\s*(?:Correct\s+Answer|$))",
        # Format: Explanation - text
        r"(?:^|\n)\s*Explanation\s*-\s*([^\n]+)",
        # Format: Explanation (with optional colon)
        r"(?:^|\n)\s*Explanation\s*:?\s*([^\n]+)"
    ]
    
    for pattern in explanation_patterns:
        # Find all matches and take the last one
        explanation_matches = list(re.finditer(pattern, cleaned_response))
        if explanation_matches:
            result['explanation'] = explanation_matches[-1].group(1).strip()
            break
    
    # Try to extract the correct answer
    answer_patterns = [
        # Format: Correct Answer: text (with or without colon)
        r"(?:^|\n)\s*Correct\s+Answer\s*:?\s*(.+?)(?:\n|$)",
        # Format: Answer: text (with or without colon)
        r"(?:^|\n)\s*Answer\s*:?\s*(.+?)(?:\n|$)",
        # Format: The answer is text
        r"(?:^|\n)\s*The\s+answer\s+is\s*:?\s*(.+?)(?:\n|$)",
        # Format: Solution: text
        r"(?:^|\n)\s*Solution\s*:?\s*(.+?)(?:\n|$)",
    ]
    
    for pattern in answer_patterns:
        answer_match = re.search(pattern, cleaned_response)
        if answer_match:
            result['correct_answer'] = answer_match.group(1).strip()
            break
    
    # If we couldn't find the answer in the cleaned response, try the original response
    # This is a fallback in case the cleaning process affected the answer patterns
    if not result['correct_answer']:
        for pattern in answer_patterns:
            answer_match = re.search(pattern, response)
            if answer_match:
                result['correct_answer'] = answer_match.group(1).strip()
                break
    
    # Final validation
    if not result['question'] or not result['correct_answer'] or not result['explanation']:
        return None

    # Additional validation for correct_answer
    if result['correct_answer'].strip() == "" or result['explanation'].strip() == "" or result['question'].strip() == "":
        return None

    return result


def parse_abcd(response: str) -> str:
    """
    Parses an LLM response to extract a single character answer (A, B, C, or D).
    
    Args:
        response (str): The full response text from the LLM
        
    Returns:
        str: A single character (A, B, C, or D) or None if no valid answer is found
    """
    # Strip whitespace and get only the first character
    cleaned_response = response.strip()
    
    # If the response is empty, return None
    if not cleaned_response:
        return None
    
    # Check for common patterns
    # Pattern 1: Just the letter
    if len(cleaned_response) == 1 and cleaned_response.upper() in "ABCD":
        return cleaned_response.upper()
    
    # Pattern 2: Letter with punctuation (A., A:, A), (A))
    match = re.match(r"^([A-D])[.):,]?$", cleaned_response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Pattern 3: "The answer is X" or similar phrases
    match = re.search(r"(?:answer|option|choice)(?:\s+is)?[\s:]*([A-D])(?:[.)]|\b)", 
                     cleaned_response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # If we get here, try to find any A, B, C, or D in the response
    letters = re.findall(r"\b([A-D])\b", cleaned_response, re.IGNORECASE)
    if letters:
        # Return the first valid letter found
        return letters[0].upper()
    
    # No valid answer found
    return None


def parse_number(response: str, valid_options: list = None) -> int:
    """
    Parses an LLM response to extract a single number.
    
    Args:
        response (str): The full response text from the LLM
        valid_options (list, optional): List of valid integer options. If provided,
                                        the function will prioritize returning a number
                                        from this list.
        
    Returns:
        int: The extracted number or None if no valid number is found
    """
    if not response or not isinstance(response, str):
        return None
        
    # Clean the response
    cleaned_response = response.strip()
    if not cleaned_response:
        return None
    
    # Look for "ANSWER: <number>" pattern
    answer_match = re.search(r'(?:ANSWER|Answer|answer):\s*(\d+)', cleaned_response)
    if answer_match:
        try:
            num = int(answer_match.group(1))
            # If valid_options is provided, check if this number is valid
            if valid_options is None or num in valid_options:
                return num
        except ValueError:
            pass
    
    # Try to extract a number using regex
    # Look for digits, possibly with commas, periods, or surrounded by text
    matches = re.findall(r'(?:^|[^\d])(\d+(?:,\d{3})*(?:\.\d+)?)(?:[^\d]|$)', cleaned_response)
    
    extracted_numbers = []
    for match in matches:
        # Remove commas and convert to int
        number_str = match.replace(',', '')
        try:
            # Handle potential decimal values by truncating
            num = int(float(number_str))
            extracted_numbers.append(num)
        except ValueError:
            pass
    
    # Try direct conversion as fallback
    try:
        num = int(cleaned_response)
        extracted_numbers.append(num)
    except ValueError:
        # Look for number words
        number_words = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 
            'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
        }
        
        lower_response = cleaned_response.lower()
        for word, value in number_words.items():
            if word in lower_response:
                extracted_numbers.append(value)
    
    # If valid_options is provided, prioritize numbers that are in valid_options
    if valid_options and extracted_numbers:
        for num in extracted_numbers:
            if valid_options is None or num in valid_options:
                return num
    
    # If no valid options match or valid_options is None, return the first extracted number
    if extracted_numbers:
        num = extracted_numbers[0]
        # If valid_options is provided, check if this number is valid
        if valid_options is None or num in valid_options:
            return num
                
    return None
