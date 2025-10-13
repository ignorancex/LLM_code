import os
import argparse
import json
import ast
import pickle
from typing import List, Tuple

import numpy as np
import torch
from mmengine.config import Config
import SurgVLP.surgvlp as surgvlp


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def load_string_list(filepath: str) -> List[str]:
    """
    Loads a list of strings from a .json file which may contain:
      - a JSON array (["a", "b", ...])
      - a JSON object with a key holding the list (e.g., {"concepts": [...]}). If multiple
        list-like values exist, the longest list is chosen.
      - a Python list literal text (e.g., "['a', 'b']").
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    # 1) Try strict JSON
    try:
        obj = json.loads(raw)
        if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
            return obj
        if isinstance(obj, dict):
            # pick the longest string-list value
            candidates = []
            for v in obj.values():
                if isinstance(v, list) and all(isinstance(x, str) for x in v):
                    candidates.append(v)
            if candidates:
                candidates.sort(key=len, reverse=True)
                return candidates[0]
    except json.JSONDecodeError:
        pass

    # 2) Try Python literal (e.g., "['a','b']")
    try:
        lit = ast.literal_eval(raw)
        if isinstance(lit, list) and all(isinstance(x, str) for x in lit):
            return lit
    except Exception:
        pass

    # 3) Fallback: one string per line (ignore empties)
    lines = [ln.strip() for ln in raw.splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        raise ValueError(f"Could not parse a list of strings from {filepath}")
    return lines


def batched(iterable, batch_size: int):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]


def extract_text_embeddings(
    texts: List[str],
    model,
    device: torch.device,
    batch_size: int = 512
) -> np.ndarray:
    """
    Tokenize with surgvlp.tokenize and encode via model(None, tokens, mode='text').
    Returns L2-normalized float32 embeddings of shape (N, D).
    """
    all_vecs = []
    model.eval()
    with torch.no_grad():
        for chunk in batched(texts, batch_size):
            tokens = surgvlp.tokenize(chunk, device=device)  # framework-provided tokenizer
            out = model(None, tokens, mode='text')           # {'text_emb': (B, D)}
            text_emb = out['text_emb']
            # L2 normalize
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            all_vecs.append(text_emb.detach().cpu())
    emb = torch.cat(all_vecs, dim=0).numpy().astype(np.float32)
    return emb


def main():
    parser = argparse.ArgumentParser(description="Extract PeskaVLP text features from concept sets")
    parser.add_argument("--config", type=str, default="SurgVLP/tests/config_peskavlp.py", help="dataset/model config file")
    parser.add_argument("--concept_dir", type=str, default="./concept_sets", help="directory containing *.json")
    parser.add_argument("--out_dir", type=str, default="./extracted_features/text", help="output directory for .pkl")
    parser.add_argument("--batch_size", type=int, default=512, help="batch size for tokenization/encoding")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(args.out_dir)

    # Load model once
    configs = Config.fromfile(args.config, lazy_import=False)['config']
    model, _ = surgvlp.load(configs.model_config)
    model = model.to(device)
    model.eval()

    # Gather input files
    files = [f for f in os.listdir(args.concept_dir) if f.lower().endswith(".json")]
    files.sort()
    if not files:
        print(f"[WARN] No *.json found in {os.path.abspath(args.concept_dir)}")
        return

    print(f"[INFO] Found {len(files)} files under {args.concept_dir}")
    for fname in files:
        in_path = os.path.join(args.concept_dir, fname)
        base = os.path.splitext(fname)[0]
        out_path = os.path.join(args.out_dir, f"{base}.pkl")

        try:
            concepts = load_string_list(in_path)
            if len(concepts) == 0:
                print(f"[SKIP] {fname}: empty list")
                continue

            emb = extract_text_embeddings(concepts, model, device, batch_size=args.batch_size)

            payload = {
                "concepts": concepts,
                "embeddings": emb,                 # shape: (N, D), float32, L2-normalized
                "emb_dim": int(emb.shape[1]),
                "model_config": str(configs.model_config),
            }
            with open(out_path, "wb") as f:
                pickle.dump(payload, f)

            print(f"[OK] {fname} -> {out_path} (N={len(concepts)}, D={emb.shape[1]})")

        except Exception as e:
            print(f"[ERROR] {fname}: {e}")

    print("[DONE] All files processed.")


if __name__ == "__main__":
    main()
