# hf_dataset_viewer.py
"""
Hugging Face Dataset Viewer (Image‑centric)
-----------------------------------------
A lightweight **Streamlit** web‑app for scrolling through any Hugging Face dataset
containing images (e.g. ``AdaptLLM/biomed‑VQA‑benchmark``) one sample per row.

Key UI improvements (v4)
~~~~~~~~~~~~~~~~~~~~~~~~
* **Dual image‑field support** – gracefully handles datasets that have either
  a single ``image`` field **or** a multi‑image ``images`` field (sequence of
  ``datasets.Image``). If both somehow appear, the row is flagged.
* **Deprecation fixed** – uses ``use_container_width`` (Streamlit ≥1.35).
* **Callback‑driven pagination** – no more ``experimental_rerun`` tracebacks.

Usage
~~~~~
```bash
pip install streamlit datasets pillow
streamlit run hf_dataset_viewer.py
```
Then open the printed URL (default http://localhost:8501).
"""
from __future__ import annotations

import dotenv

dotenv.load_dotenv(override=True)  # Load .env if present, e.g. for HF_TOKEN


from typing import Any, Dict, List

import streamlit as st
from datasets import load_dataset
from PIL import Image  # noqa: F401 – type hints & safety

###############################################################################
# Page & sidebar configuration
###############################################################################
st.set_page_config(page_title="HF Dataset Viewer", layout="wide")

st.sidebar.title("📚 Dataset Settings")
DATASET_NAME: str = st.sidebar.text_input(
    "Dataset name",
    value="AdaptLLM/biomed-VQA-benchmark",
    help="<namespace>/<repo> on the Hub",
)
SUBSET: str = st.sidebar.text_input("Subset (blank if none)", value="PMC-VQA")
SPLIT: str = st.sidebar.text_input("Split", value="test")
PAGE_SIZE: int = st.sidebar.number_input(
    "Rows per page", min_value=1, max_value=100, value=10
)


###############################################################################
# Helpers – cached dataset loader
###############################################################################
@st.cache_resource(
    show_spinner=False, hash_funcs={"datasets.arrow_dataset.Dataset": lambda _: None}
)
def get_dataset(name: str, subset: str, split: str):
    """Load and cache the requested dataset split."""
    if subset.strip():
        return load_dataset(name, subset, split=split)
    return load_dataset(name, split=split)


###############################################################################
# Dataset loading trigger
###############################################################################
if st.sidebar.button("🚀 Load / Reload dataset") or "dataset" not in st.session_state:
    with st.spinner("Preparing dataset …"):
        st.session_state.dataset = get_dataset(DATASET_NAME, SUBSET, SPLIT)
        st.session_state.page = 0

###############################################################################
# Pagination callbacks (automatic rerun)
###############################################################################
if "dataset" in st.session_state:
    ds = st.session_state.dataset
    total = len(ds)
    pages = (total + PAGE_SIZE - 1) // PAGE_SIZE
    page = st.session_state.get("page", 0)

    def prev_page():
        st.session_state.page = max(st.session_state.page - 1, 0)

    def next_page():
        st.session_state.page = min(st.session_state.page + 1, pages - 1)

    # Sidebar pagination controls
    st.sidebar.divider()
    st.sidebar.write(f"Page **{page + 1}/{pages}** of {total} samples")
    col_prev, col_next = st.sidebar.columns(2)
    with col_prev:
        st.button("⬅ Prev", disabled=page == 0, on_click=prev_page)
    with col_next:
        st.button("Next ➡", disabled=page >= pages - 1, on_click=next_page)

    # Sample slice bounds
    start, end = page * PAGE_SIZE, min((page + 1) * PAGE_SIZE, total)

    st.divider()
    for idx in range(start, end):
        sample: Dict[str, Any] = ds[idx]

        # Determine image mode -------------------------------------------------
        has_single = "image" in sample and sample["image"] is not None
        has_multi = "images" in sample and sample["images"] is not None

        col_img, col_meta = st.columns([1, 4], gap="large")
        with col_img:
            if has_single and not has_multi:
                # Single image case
                st.image(sample["image"], use_container_width=True)
            elif has_multi and not has_single:
                # Multiple images – ``images`` is typically a list/sequence of PIL
                imgs: List[Image.Image] = sample["images"]  # type: ignore[assignment]
                if len(imgs) == 1:
                    st.image(imgs[0], use_container_width=True)
                else:
                    st.image(imgs, use_container_width=True)  # Streamlit stacks them
            elif has_single and has_multi:
                st.warning(
                    "Sample has both 'image' and 'images' keys – displaying single."
                )
                st.image(sample["image"], use_container_width=True)
            else:
                st.write("*(no image)*")

        # Metadata (exclude image keys) ---------------------------------------
        with col_meta:
            st.markdown(f"**Index:** {idx}")
            for k, v in sample.items():
                if k in {"image", "images"}:
                    continue
                vc = (
                    v
                    if not isinstance(v, str)
                    else v[:500] + ("…" if len(v) > 500 else "")
                )
                st.markdown(f"**{k}:** {vc}")
        st.divider()
else:
    st.info("Configure the dataset in the sidebar and click *Load dataset*.")
