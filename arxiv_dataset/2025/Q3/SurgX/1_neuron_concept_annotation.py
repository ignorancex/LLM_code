import os
import json
import pickle
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F


def load_any_pkl(path: str):
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


def normalize_torch(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return F.normalize(x, p=2, dim=dim, eps=eps)


def compute_representative_concepts(
    representative_sequences: List[torch.Tensor],
    concept_names: List[str],
    concept_emb_np: np.ndarray,
    save_json_path: str = "representative_concepts_by_neuron.json",
) -> List[Dict[str, Any]]:
    """
    representative_sequences: 뉴런별 텐서 리스트, 각 텐서는 (num_features, 768)
    concept_names: 길이 C (=270) 컨셉 문자열 리스트
    concept_emb_np: (C, 768) float32 numpy array
    """
    
    C = len(concept_names)
    assert concept_emb_np.ndim == 2 and concept_emb_np.shape[0] == C, "concept_emb shape mismatch"
    D = concept_emb_np.shape[1]

    # 개념 임베딩 (L2 정규화)
    concept_emb = torch.from_numpy(concept_emb_np.astype(np.float32))
    concept_emb = normalize_torch(concept_emb, dim=1)  # (C, D)

    results: List[Dict[str, Any]] = []

    for n_idx, feats in enumerate(representative_sequences):
        if isinstance(feats, np.ndarray):
            feats = torch.from_numpy(feats)
        feats = feats.float().contiguous()  # (F, D) or (0, D)

        if feats.numel() == 0:
            results.append({
                "neuron_idx": n_idx,
                "num_features": 0,
                "threshold": None,
                "min": None,
                "max": None,
                "selected_count": 0,
                "selected_concepts_sorted": [],  # name & score 저장
                "selected_indices_sorted": [],
                "selected_scores_sorted": [],
            })
            print(f"[neuron {n_idx:03d}] empty features -> selected 0 concepts")
            continue

        assert feats.shape[1] == D, f"feature dim mismatch at neuron {n_idx}: {feats.shape} vs D={D}"

        # feature 정규화
        feats = normalize_torch(feats, dim=1)  # (F, D)

        # 코사인 유사도 (F x C) = (F x D) @ (D x C)
        sims = feats @ concept_emb.t()  # (F, C)

        # 컨셉별 평균 유사도
        avg_sim = sims.mean(dim=0)  # (C,)

        vmin = float(avg_sim.min().item())
        vmax = float(avg_sim.max().item())
        thr = vmax - 0.05 * (vmax - vmin)  # max - 0.1*(max-min)

        sel_mask = avg_sim >= thr
        sel_idx = sel_mask.nonzero(as_tuple=False).squeeze(-1).tolist()

        # ✅ 선택된 컨셉들을 평균 cos sim 내림차순으로 정렬
        sel_idx_sorted = sorted(sel_idx, key=lambda i: float(avg_sim[i].item()), reverse=True)
        sel_scores_sorted = [float(avg_sim[i].item()) for i in sel_idx_sorted]
        sel_concepts_sorted = [
            {"index": i, "name": concept_names[i], "avg_cos_sim": float(avg_sim[i].item())}
            for i in sel_idx_sorted
        ]

        # (옵션) 전체 컨셉을 점수순으로 저장하고 싶으면 아래 주석 해제
        # all_idx_sorted = torch.argsort(avg_sim, descending=True).tolist()
        # all_concepts_sorted = [
        #     {"index": i, "name": concept_names[i], "avg_cos_sim": float(avg_sim[i].item())}
        #     for i in all_idx_sorted
        # ]

        results.append({
            "neuron_idx": n_idx,
            "num_features": int(feats.shape[0]),
            "threshold": thr,
            "min": vmin,
            "max": vmax,
            "selected_count": len(sel_idx_sorted),
            "selected_concepts_sorted": sel_concepts_sorted,  # ← 이름+점수 포함
            "selected_indices_sorted": sel_idx_sorted,        # ← 정렬된 인덱스
            "selected_scores_sorted": sel_scores_sorted,      # ← 정렬된 점수
            # "all_concepts_sorted": all_concepts_sorted,     # ← 필요 시 활성화
        })

        top1_name = sel_concepts_sorted[0]["name"] if sel_concepts_sorted else "—"
        top1_score = sel_concepts_sorted[0]["avg_cos_sim"] if sel_concepts_sorted else float("nan")
        print(f"[neuron {n_idx:03d}] F={feats.shape[0]} | min={vmin:.4f} max={vmax:.4f} thr={thr:.4f} "
              f"| selected={len(sel_idx_sorted)} | top1={top1_name}({top1_score:.4f})")
    os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
    # 저장
    with open(save_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[DONE] Saved representative concept sets -> {save_json_path}")

    return results


if __name__ == "__main__":
    spr_model = "ASFormer"
    selection_method = "4_Video_wise_Top1"
    concept_sets = "ChoLec-270"
    
    concept_pkl = f"extracted_features/{spr_model}/text/{concept_sets}.pkl"
    concept_payload = load_any_pkl(concept_pkl)
    concept_names: List[str] = concept_payload["concepts"]              # 길이 270
    concept_emb_np: np.ndarray = concept_payload["embeddings"]          # (270, 768)

    # 2) 대표 시퀀스 feature 리스트 로드
    seq_pkl = f"extracted_features/{spr_model}/sequence/{selection_method}.pkl"
    representative_sequences = load_any_pkl(seq_pkl)  # list of tensors (F_i, 768)

    # 3) 계산 및 저장
    _ = compute_representative_concepts(
        representative_sequences=representative_sequences,
        concept_names=concept_names,
        concept_emb_np=concept_emb_np,
        save_json_path=f"extracted_neuron_concepts/{spr_model}/{selection_method}_and_{concept_sets}.json",
    )
