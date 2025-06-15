import json
import subprocess
import tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def is_valid_cpp(code: str, compiler="g++", std="c++17") -> bool:
    """
    Write `code` to a temp .cpp file and invoke the compiler in syntax-checking mode.
    Returns True if compilation succeeds (exit code 0), False otherwise.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cpp", delete=False) as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    proc = subprocess.run(
        [compiler, f"-std={std}", "-fsyntax-only", tmp_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    return proc.returncode == 0

def validate_item(item):
    """
    Returns the original item if both code blocks compile cleanly,
    or None otherwise.
    """
    code_block = item.get('generate_code_block', '').strip()
    ref_block  = item.get('generate_ref_code_block', '').strip()
    if not (code_block and ref_block):
        return None

    if is_valid_cpp(code_block) and is_valid_cpp(ref_block):
        return item
    return None

def main(model="DeepSeek", max_workers=4):
    input_path  = Path(f'LLM_code/codeforces/simulation/temp/{model}_cpp.json')
    output_path = Path(f'LLM_code/codeforces/simulation/output/{model}_cpp.json')

    data = json.loads(input_path.read_text(encoding='utf-8'))
    valid_data = []

    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(validate_item, itm): itm for itm in data}
        for fut in tqdm(as_completed(futures),
                        total=len(futures),
                        desc=f"Validating C++ for {model}",
                        unit="file"):
            result = fut.result()
            if result is not None:
                valid_data.append(result)

    output_path.write_text(json.dumps(valid_data, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"✅ {len(valid_data)} valid C++ submissions retained.")

if __name__ == "__main__":
    # adjust max_workers to number of CPU cores you want to use
    main(model="Qwen", max_workers=16)
