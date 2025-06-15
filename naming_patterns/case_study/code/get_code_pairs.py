import json

def find_code_with_specific_variable_presence(json_file_path):
    """
    Analyzes a JSON file for specific variable presence conditions in code fields.

    Args:
        json_file_path (str): The path to the JSON file.
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            print(f"Warning: The JSON file '{json_file_path}' does not contain a list of objects at the root. Processing might not be as expected.")
            # If it's a single object, put it in a list to iterate
            data = [data]

        found_count = 0
        print("--- Matching Entries ---")
        for entry in data:
            source_code = entry.get('sourceCode', '')
            generate_ref_code_block = entry.get('generate_ref_code_block', '')

            # Check if 'max_length' is NOT in source_code AND is IN generate_ref_code_block
            if 'max_length' in generate_ref_code_block and 'max_length' not in source_code:
                found_count += 1
                print(f"\nEntry {found_count}:")
                print("Source Code:")
                print(source_code)
                print("\nGenerate Ref Code Block:")
                print(generate_ref_code_block)
                print("-" * 30) # Separator for readability

        if found_count == 0:
            print("No entries found matching the criteria.")

    except FileNotFoundError:
        print(f"Error: The file '{json_file_path}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{json_file_path}'. Please check the file format.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- How to use the script ---
if __name__ == "__main__":
    # Replace 'your_json_file.json' with the actual path to your JSON file
    json_file = 'LLM_code/dataset_collection/simulation/output/Qwen_python.json'
    find_code_with_specific_variable_presence(json_file)