import ctypes
from pathlib import Path

def load_from_clib(relative_function_path: str):
    fp = Path(__file__).parent.parent.parent / "clib" / relative_function_path
    return ctypes.CDLL(str(fp))