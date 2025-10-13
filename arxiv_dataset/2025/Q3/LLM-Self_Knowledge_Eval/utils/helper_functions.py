import random, string      

def generate_id(length=8):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def find_outer_key(data, inner_key):
    for outer_key, inner_dict in data.items():
        if inner_key in inner_dict:
            return outer_key  
    return None  
