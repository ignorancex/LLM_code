import os
import tempfile

def file_writer(code: str) -> str:
    """
    Write Prolog code to a temporary file and return the file path.

    Args:
        code (str): The Prolog code to write.

    Returns:
        str: Path to the written temporary file.
    """
    temp_dir = os.path.join(os.getcwd(), "DATA", "TEMP")
    os.makedirs(temp_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=temp_dir, suffix=".pl") as temp_file:
        temp_file.write(code.encode())
        return temp_file.name
