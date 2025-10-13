import re

def normalize_string(s):
    # Remove leading/trailing whitespace
    s = s.strip()

    # Remove quotation marks (both single and double quotes)
    s = s.replace('"', '').replace("'", "")

    # Convert to lowercase to make comparison case-insensitive
    s = s.lower()

    # Optionally, remove other special characters (if needed)
    s = re.sub(r'[^\w\s]', '', s)

    return s

def trim_response(prompt, response):
    # Try to match the ending of the prompt with the beginning of the response
    match = re.search(re.escape(prompt), response)
    if match:
        return response[match.end():]
    return response

def extract_choices_and_intro(text):
    # Regular expression to match "Choice X:" and its variants
    # Handle optional "#", spaces, and various punctuation
    pattern = r"Choice\s*#?\s*\d+[:.]?\s*.*?(?=\s*Choice\s*#?\s*\d+[:.]?|$)"

    # Extract everything before the first "Choice X:"
    intro_pattern = r"(.*?)(?=\s*Choice\s*#?\s*\d+[:.]?)"

    # Use re.IGNORECASE to make the pattern case-insensitive
    intro_match = re.search(intro_pattern, text, re.DOTALL | re.IGNORECASE)

    if intro_match:
        intro = intro_match.group(0).strip()  # Get the introduction, remove leading/trailing spaces
    else:
        intro = text.strip()  # If no "Choice" is found, treat the whole text as the intro

    # Find all matches for the choices in the text
    choices = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)

    return intro, choices
