import sys
from datasets import load_dataset

cache_directory = sys.argv[1] if len(sys.argv) > 1 else None

persona_chat = load_dataset("bavard/personachat_truecased", cache_dir=cache_directory)
qnli = load_dataset("glue", "qnli", cache_dir=cache_directory)
