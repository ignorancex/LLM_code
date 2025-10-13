#!/usr/bin/env python3
# coding: utf-8

"""
Orchestrate the feature extraction and reduction pipeline for music and painting.
Runs the following steps sequentially:
- Music: feature_extraction_music.py → reduce_normalize_music.py
- Painting: emotion_preprocess.py → feature_extraction_painting.py → reduce_normalize_painting.py

Author:
- Bereket A. Yilma <name.surname@artaicare.com>

Dependencies: Install all requirements from requirements.txt (`pip install -r requirements.txt`).
"""

import os
import subprocess
import sys


def run_script(script_path):
    """Run a Python script and handle its execution."""
    try:
        print(f"Running {script_path}...")
        result = subprocess.run(
            [sys.executable, script_path], check=True, text=True, capture_output=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}: {e.stderr}")
        raise
    except Exception as e:
        print(f"Unexpected error running {script_path}: {str(e)}")
        raise


if __name__ == "__main__":
    # Script paths
    project_root = os.path.dirname(os.path.abspath(__file__))
    music_extraction = os.path.join(
        project_root, "feature_extraction", "music", "feature_extraction_music.py"
    )
    music_reduction = os.path.join(
        project_root, "feature_extraction", "music", "reduce_normalize_music.py"
    )
    painting_preprocess = os.path.join(
        project_root, "feature_extraction", "painting", "emotion_preprocess.py"
    )
    painting_extraction = os.path.join(
        project_root, "feature_extraction", "painting", "feature_extraction_painting.py"
    )
    painting_reduction = os.path.join(
        project_root, "feature_extraction", "painting", "reduce_normalize_painting.py"
    )

    # Run the pipelines
    print("Starting music feature pipeline...")
    run_script(music_extraction)
    run_script(music_reduction)

    print("\nStarting painting feature pipeline...")
    run_script(painting_preprocess)
    run_script(painting_extraction)
    run_script(painting_reduction)

    print("\nFeature extraction and reduction pipeline completed successfully!")

    # llm_vlm scripts
    music_multi_modal = os.path.join(
        project_root, "feature_extraction", "llm_vlm", "multi_modal_music.py"
    )
    painting_multi_modal = os.path.join(
        project_root, "feature_extraction", "llm_vlm", "multi_modal_painting.py"
    )
    similarity_computation = os.path.join(
        project_root, "feature_extraction", "llm_vlm", "similarity_computation.py"
    )

    print("\nStarting LLM/VLM music feature processing...")
    run_script(music_multi_modal)

    print("\nStarting LLM/VLM painting feature processing...")
    run_script(painting_multi_modal)

    print("\nComputing LLM/VLM cross-modal similarity...")
    run_script(similarity_computation)

    print("\nLLM/VLM processing completed successfully!")
