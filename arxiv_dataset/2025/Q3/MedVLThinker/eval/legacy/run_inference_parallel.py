#!/usr/bin/env python
"""
run_med_vqa.py ― parallel version

Query a running vLLM (OpenAI‑compatible) server with Qwen‑2.5‑VL
(or another VLM) and evaluate predictions on a medical VQA dataset.

Example:
python run_med_vqa.py \
    --dataset-name AdaptLLM/biomed-VQA-benchmark \
    --subset PMC-VQA \
    --split test \
    --base-url http://localhost:8000/v1 \
    --model qwen2.5-vl-7b-instruct \
    --out-file preds_pmc_test.json \
    --num-workers 16
"""
import dotenv

dotenv.load_dotenv(override=True)
import base64
import io
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import backoff
import click
import openai
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


# ------------------------- helper functions ------------------------- #
def pil_to_base64(img: Image.Image, fmt: str = "JPEG") -> str:
    """Encode PIL image to base64 string suitable for OpenAI vision input."""
    buf = io.BytesIO()
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(buf, format=fmt)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def build_messages(
    image_b64: str,
    question: str,
    options: Dict[str, str],
    system_prompt: str,
    cot: bool,
) -> List[Dict[str, Any]]:
    """Construct OpenAI‑vision chat messages with optional CoT."""
    prompt_lines = [
        "You should provide your thoughts within <think> </think> tags, then "
        "answer with just one of the options below within <answer> </answer> "
        "tags (For example, if the question is \n’Is the earth flat?\n "
        "A: Yes\nB: No’, you should answer with "
        "<think>...</think> <answer>B: No</answer>)."
    ]
    qtxt = f"Question: {question}\nOptions:"
    for letter in ["A", "B", "C", "D"]:
        qtxt += f"\n{letter}. {options[letter]}"
    if cot:
        qtxt += "\n\nLet's think step by step."

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
                {"type": "text", "text": "\n".join(prompt_lines) + "\n\n" + qtxt},
            ],
        },
    ]


def extract_answer(text: str) -> str:
    """Extract the model’s final option (A/B/C/D) from its output."""
    m = re.search(r"<answer>(.*?)</answer>", text, re.S)
    return (m.group(1).strip() if m else text.strip())[:1].upper()


def create_chat_completion(client, **kwargs) -> str:
    """
    Call client.chat.completions.create(**kwargs) with automatic retries.
    Returns the trimmed text string when successful.
    backoff handles the sleep + retry; it re‑raises on final failure.
    """
    resp = client.chat.completions.create(**kwargs)
    return resp


# ------------------------------- CLI -------------------------------- #
@click.command()
@click.option("--dataset-name", default="AdaptLLM/biomed-VQA-benchmark")
@click.option("--subset", default="PMC-VQA")
@click.option("--split", default="test")
@click.option("--base-url", default="http://localhost:8000/v1")
@click.option("--model", "model_name", default="qwen2.5-vl-7b-instruct")
@click.option("--out-file", default="preds.json")
@click.option("--cot/--no-cot", default=True)
@click.option("--dataset-size", type=int, default=None, help="Debug subset size.")
@click.option(
    "--num-workers",
    type=int,
    default=8,
    show_default=True,
    help="Concurrent request workers.",
)
@click.option(
    "--num_proc", type=int, default=16, help="Number of processes for dataset loading."
)
def main(
    dataset_name: str,
    subset: str,
    split: str,
    base_url: str,
    model_name: str,
    out_file: str,
    cot: bool,
    dataset_size: int,
    num_workers: int,
    num_proc: int,
):
    # ---------- OpenAI client (thread‑safe) ----------
    openai_api_key = "EMPTY"  # arbitrary for local vLLM
    openai_api_base = base_url
    client = openai.OpenAI(api_key=openai_api_key, base_url=openai_api_base)

    # ---------- Load dataset ----------
    ds = load_dataset(dataset_name, subset)[split]
    if dataset_size:
        ds = ds.select(range(dataset_size))

    # TODO: resize image if needed
    ds = ds.map(
        resize_image_to_qwen2_5_vl,
        num_proc=num_proc,
        desc="Resizing images to fit Qwen2.5-VL requirements",
    )

    # ---------- Worker ‑ one example ----------
    def process_example(pair: Tuple[int, dict]) -> dict:
        idx, ex = pair
        try:
            img_b64 = pil_to_base64(ex["image"])
            opts = {k: ex[k] for k in ["A", "B", "C", "D"]}
            messages = build_messages(
                img_b64, ex["input"], opts, system_prompt="", cot=cot
            )
            resp = create_chat_completion(
                client=client,
                model=model_name,
                messages=messages,
                temperature=0.2,
                # max_tokens=2048,
                max_completion_tokens=2048,
            )
            output_text = resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error processing example {idx}: {e}, idx {idx}")
            output_text = f"<<ERROR: {e}>>"

        pred_letter = extract_answer(output_text)
        return {
            "id": idx,
            "question": ex["input"],
            "options": opts,
            "label": ex["label"],
            "prediction": output_text,
            "pred_letter": pred_letter,
            "correct": pred_letter == ex["label"],
        }

    # ---------- Parallel inference ----------
    results: List[dict] = []
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = [pool.submit(process_example, (i, ex)) for i, ex in enumerate(ds)]
        for _ in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Running inference",
            unit="sample",
        ):
            pass  # tqdm handles progress; results gathered below

        # Collect in the order they finished
        for fut in futures:
            results.append(fut.result())

    # Re‑sort to the original dataset order (optional but convenient)
    results.sort(key=lambda r: r["id"])

    # ---------- Save & report ----------
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    acc = sum(r["correct"] for r in results) / len(results)
    print(f"\nSaved {len(results)} records to '{out_file}'. " f"Accuracy: {acc:.2%}")


# https://github.com/Yuxiang-Lai117/Med-R1/blob/53e46ba24e04d7d7705db4750551786a93e96493/src/r1-v/local_scripts/prepare_hf_data.py#L144
def resize_image_to_qwen2_5_vl(example):
    # for Qwen2-VL-2B's processor requirement
    # Assuming the image is in a format that can be checked for dimensions
    # You might need to adjust this depending on how the image is stored in your dataset
    image = example["image"]
    if image.height < 28 or image.width < 28:
        image = resize_shortest_dim(image, target_size=28)
    example["image"] = image

    if image.height > 1024 or image.width > 1024:
        image = resize_longest_dim(image, target_size=1024)
        example["image"] = image

    return example


def resize_shortest_dim(image, target_size=28):
    """Resize the shortest dimension of the image to the specified size."""
    width, height = image.size
    if width < height:
        new_width = target_size
        new_height = int(height * (target_size / width))
    else:
        new_height = target_size
        new_width = int(width * (target_size / height))
    return image.resize((new_width, new_height), Image.BILINEAR)


# resize the longest dim to 384
def resize_longest_dim(image, target_size=1024):
    width, height = image.size
    if width > height:
        new_width = target_size
        new_height = int(height * (target_size / width))
    else:
        new_height = target_size
        new_width = int(width * (target_size / height))
    return image.resize((new_width, new_height), Image.BILINEAR)


if __name__ == "__main__":
    main()
