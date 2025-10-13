

def str_to_bool(string: str) -> bool:
    if string.lower() == "true":
        return True
    elif string.lower() == "false":
        return False
    else:
        raise RuntimeError(f"Failed to parse string '{string}' to boolean value...")