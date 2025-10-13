def group_by_context(data: list[dict]) -> list[dict]:
    """
    Groups data entries by their context field.
    
    Args:
        data: List of dictionaries, each containing a 'context' key
        
    Returns:
        List of dictionaries with 'context' and 'questions' keys
    """
    # Use a dictionary for faster grouping
    merged_data = {}
    
    for entry in data:
        context = entry['context']
        
        # If this context is new, initialize it
        if context not in merged_data:
            merged_data[context] = []
        
        # Add entry without context field to the appropriate group
        merged_data[context].append({k: v for k, v in entry.items() if k != 'context'})
    
    # Convert to final format
    return [{'context': context, 'questions': entries} 
            for context, entries in merged_data.items()]


def flatten_by_context(data: list[dict]) -> list[dict]:
    """
    Flattens a list of dictionaries by their context field.
    
    Args:
        data: List of dictionaries, each containing a 'context' key

    Returns:
        List of dictionaries with 'context' and 'questions' keys
    """
    # Use a dictionary for faster grouping
    merged_data = []
    for entry in data:
        context = entry['context']
        if 'questions' in entry:
            for question in entry['questions']:
                dat = {}
                dat['context'] = context
                for k, v in question.items():
                    dat[k] = v
                merged_data.append(dat)
        else:
            dat = entry
            merged_data.append(dat)
        
        
    return merged_data


def remove_empty_logs(fp: str):
    import os
    import json
    fns = [os.path.join(fp, fn) for fn in os.listdir(fp) if fn.endswith('.json')]
    for fn in fns:
        with open(fn, 'r') as f:
            data = json.load(f)
            if 'results' not in data:
                os.remove(fn)

def get_completed_logs(fp: str) -> list[dict]:
    import os
    import json
    if not os.path.exists(fp):
        return [], []
    fns = [os.path.join(fp, fn) for fn in os.listdir(fp) if fn.endswith('.json')]
    fns.sort()
    completed_logs = []
    completed_fns = []
    for fn in fns:
        with open(fn, 'r') as f:
            data = json.load(f)
        if 'results' in data:
            completed_logs.append(data['eval'])
            completed_fns.append(fn)
        else:
            import os
            os.remove(fn)
    return completed_logs, completed_fns

# def is_copmleted(log_fp: str, model: str, dataset: str) -> bool:
#     import os
#     import json
#     fns = [os.path.join(log_fp, fn) for fn in os.listdir(log_fp) if fn.endswith('.json')]
#     fns = [fn for fn in fns if dataset in fn]
#     fns.sort()
#     for fn in fns:
#         with open(fn, 'r') as f:
#             log = json.load(f)
#             log = log['eval']
#         if 'results' in log:
#             if log['model'] == model and log['dataset']['name'] == dataset:
#                 return True
#     return False
