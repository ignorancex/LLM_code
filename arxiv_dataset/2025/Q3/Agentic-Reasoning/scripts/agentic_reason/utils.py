import re
from typing import Dict, Optional
from agentic_reason.config import BEGIN_SEARCH_QUERY, BEGIN_SEARCH_RESULT

def parse_steps(text: str) -> Dict[int, str]:
    """Parse reasoning steps from text into a dictionary."""
    steps = {}
    current_step = None
    current_content = []
    
    for line in text.split('\n'):
        # Try to match a step number at the start of the line
        step_match = re.match(r'^Step\s*(\d+):\s*(.*)$', line.strip())
        
        if step_match:
            # If we were building a previous step, save it
            if current_step is not None:
                steps[current_step] = '\n'.join(current_content).strip()
            
            # Start new step
            current_step = int(step_match.group(1))
            current_content = [f"Step {current_step}: {step_match.group(2)}"]
        elif current_step is not None:
            current_content.append(line)
    
    # Save the last step if exists
    if current_step is not None:
        steps[current_step] = '\n'.join(current_content).strip()
    
    return steps

# def replace_recent_steps(origin_str: str, replace_str: str) -> str:
#     """Replace recent reasoning steps in the original string with new steps."""
#     # Reference to original implementation
#     # Reference lines from run_agentic_reason.py:
    
#     # Parse the original and replacement steps
#     origin_steps = parse_steps(origin_str)
#     replace_steps = parse_steps(replace_str)
    
#     # Apply replacements
#     for step_num, content in replace_steps.items():
#         if "DELETE THIS STEP" in content:
#             # Remove the step if it exists
#             if step_num in origin_steps:
#                 del origin_steps[step_num]
#         else:
#             # Replace or add the step
#             origin_steps[step_num] = content
    
#     # Sort the steps by step number
#     sorted_steps = sorted(origin_steps.items())
    
#     # Reconstruct the reasoning steps as a single string
#     new_reasoning_steps = "\n\n".join([f"{content}" for num, content in sorted_steps])
    
#     return new_reasoning_steps

def extract_between(text: str, start_tag: str, end_tag: str) -> Optional[str]:
    """Extract text between two tags."""
    pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None

def normalize_url(url: str) -> str:
    """Normalize URL for consistent caching."""
    url = url.strip().lower()
    if url.endswith('/'):
        url = url[:-1]
    return url

def clean_snippet(snippet: str) -> str:
    """Clean snippet text by removing HTML tags and normalizing whitespace."""
    # Remove HTML tags
    clean_text = re.sub('<[^<]+?>', '', snippet)
    # Normalize whitespace
    clean_text = ' '.join(clean_text.split())
    return clean_text

def replace_recent_steps(origin_str, replace_str):
        """
        Replaces specific steps in the original reasoning steps with new steps.
        If a replacement step contains "DELETE THIS STEP", that step is removed.

        Parameters:
        - origin_str (str): The original reasoning steps.
        - replace_str (str): The steps to replace or delete.

        Returns:
        - str: The updated reasoning steps after applying replacements.
        """

        def parse_steps(text):
            """
            Parses the reasoning steps from a given text.

            Parameters:
            - text (str): The text containing reasoning steps.

            Returns:
            - dict: A dictionary mapping step numbers to their content.
            """
            step_pattern = re.compile(r"Step\s+(\d+):\s*")
            steps = {}
            current_step_num = None
            current_content = []

            for line in text.splitlines():
                step_match = step_pattern.match(line)
                if step_match:
                    # If there's an ongoing step, save its content
                    if current_step_num is not None:
                        steps[current_step_num] = "\n".join(current_content).strip()
                    current_step_num = int(step_match.group(1))
                    content = line[step_match.end():].strip()
                    current_content = [content] if content else []
                else:
                    if current_step_num is not None:
                        current_content.append(line)
            
            # Save the last step if any
            if current_step_num is not None:
                steps[current_step_num] = "\n".join(current_content).strip()
            
            return steps

        # Parse the original and replacement steps
        origin_steps = parse_steps(origin_str)
        replace_steps = parse_steps(replace_str)

        # Apply replacements
        for step_num, content in replace_steps.items():
            if "DELETE THIS STEP" in content:
                # Remove the step if it exists
                if step_num in origin_steps:
                    del origin_steps[step_num]
            else:
                # Replace or add the step
                origin_steps[step_num] = content

        # Sort the steps by step number
        sorted_steps = sorted(origin_steps.items())

        # Reconstruct the reasoning steps as a single string
        new_reasoning_steps = "\n\n".join([f"{content}" for num, content in sorted_steps])

        return new_reasoning_steps

def extract_reasoning_context(all_reasoning_steps, mind_map = None) -> str:
        if mind_map:
            truncated_prev_reasoning = mind_map.query("summarize the reasoning process, be short and clear")
            return truncated_prev_reasoning
        else:
            truncated_prev_reasoning = ""
            for i, step in enumerate(all_reasoning_steps):
                truncated_prev_reasoning += f"Step {i + 1}: {step}\n\n"

            prev_steps = truncated_prev_reasoning.split('\n\n')
            if len(prev_steps) <= 5:
                truncated_prev_reasoning = '\n\n'.join(prev_steps)
            else:
                truncated_prev_reasoning = ''
                for i, step in enumerate(prev_steps):
                    if i == 0 or i >= len(prev_steps) - 4 or BEGIN_SEARCH_QUERY in step or BEGIN_SEARCH_RESULT in step:
                        truncated_prev_reasoning += step + '\n\n'
                    else:
                        if truncated_prev_reasoning[-len('\n\n...\n\n'):] != '\n\n...\n\n':
                            truncated_prev_reasoning += '...\n\n'
            truncated_prev_reasoning = truncated_prev_reasoning.strip('\n')
            return truncated_prev_reasoning