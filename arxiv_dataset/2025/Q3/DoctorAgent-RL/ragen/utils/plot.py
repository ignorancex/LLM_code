import base64
from io import BytesIO
from typing import List
from PIL import Image
import os
import re
import html

def image_to_base64(img_array):
    """Convert numpy array to base64 string"""
    img = Image.fromarray(img_array)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def save_trajectory_to_output(trajectory, save_dir) -> List[str]:
    """
    Save the trajectory to HTML files with better multi-language support
    
    Arguments:
        - trajectory (list): The trajectory to save
        - save_dir (str): Directory to save the HTML files
    
    Returns:
        - filenames (list): List of saved HTML file paths
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # CSS styles as a separate string
    css_styles = '''
        body {
            font-family: "Noto Sans CJK SC", "Noto Sans CJK JP", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .trajectory-step {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .image-container {
            display: flex;
            justify-content: space-between;
            margin-bottom: 15px;
        }
        .image-box {
            width: 48%;
        }
        .image-box img {
            width: 100%;
            height: auto;
            border-radius: 4px;
        }
        .image-title {
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
        }
        .response-box {
            background: #f8f8f8;
            border-radius: 4px;
            padding: 15px;
            margin: 10px 0;
            white-space: pre-wrap;
            font-size: 14px;
            line-height: 1.5;
        }
        .step-number {
            font-size: 18px;
            font-weight: bold;
            color: #2196F3;
            margin-bottom: 15px;
        }
    '''
    
    filenames = []
    for data_idx, data in enumerate(trajectory):
        steps_html = []
        n_steps = len(data['state'])
        
        for step in range(n_steps):
            if True: # TODO: how to check if the state is a string or a image?
                # Handle the state as a string
                image_state = data['state'][step]
                is_string_state = True
            else:
                # Convert images to base64
                image_state = image_to_base64(data['state'][step])
                is_string_state = False
            
            # Process response text
            parsed_response = data['parsed_response'][step]['raw']
            parsed_response = parsed_response.replace("<|im_end|>", "").replace("<|endoftext|>", "")
            parsed_response = parsed_response.replace('\\n', '\n')
            parsed_response = re.sub(r'(</think>)\s*(<answer>)', r'\1\n\2', parsed_response)
            parsed_response = html.escape(parsed_response)
            
            # Create step HTML
            if is_string_state:
                state_html = f'<div class="image-title">State</div><div class="response-box">{html.escape(image_state)}</div>'
            else:
                state_html = f'<div class="image-title">State</div><img src="data:image/png;base64,{image_state}" alt="State">'
            
            step_html = f'''
            <div class="trajectory-step">
                <div class="step-number">Step {step + 1}</div>
                <div class="image-container">
                    <div class="image-box">
                        {state_html}
                    </div>
                </div>
                <div class="response-box">
                    Model Response:
                    {parsed_response}
                </div>
            </div>
            '''
            steps_html.append(step_html)
        
        # Combine all content into final HTML
        final_html = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                {css_styles}
            </style>
        </head>
        <body>
            {''.join(steps_html)}
        </body>
        </html>
        '''
        
        # Save to file
        filename = os.path.join(save_dir, f"trajectory_data_{data_idx}.html")
        filenames.append(filename)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(final_html)
    return filenames
        
        

def parse_llm_output(llm_output: str, strategy: str):
    """
    Parse the llm output
    =========================
    Arguments:
        - llm_output: the output of the llm
        - strategy: the strategy to parse the llm output
            - "formated": parse action and thought
            - "raw": parse the raw output
    =========================
    Returns (dict): parsed output
    """
    if strategy == "raw":
        return {'raw': llm_output}
    elif strategy == "formatted":
        if "<answer>" in llm_output:
            action = llm_output.split("<answer>")[1].split("</answer>")[0].strip()
        else:
            action = ""
        if "<think>" in llm_output:
            thought = llm_output.split("<think>")[1].split("</think>")[0].strip()
        else:
            thought = ""
        return {
            'action': action,
            'thought': thought
        }
    else:
        raise ValueError(f"Unknown strategy: {strategy}")