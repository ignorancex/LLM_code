#!/usr/bin/env python3
"""
Verify README.md arXiv ID matches CITATION.cff's preferred-citation arXiv ID.
Assumes script lives at dev/tools/scripts/ and repo root is parents[2].
Handles optional vN suffix on arXiv IDs.
"""
import re
import sys
import yaml
import pathlib

try:
    repo_root = pathlib.Path(__file__).resolve().parents[3]  # Go up three levels from dev/tools/scripts to root
    readme_path = repo_root / "README.md"
    if not readme_path.is_file():
        print(f"Error: README.md not found: {readme_path}", file=sys.stderr)
        sys.exit(1)
    readme_text = readme_path.read_text(encoding="utf-8", errors="ignore")

    m = re.search(r'arXiv[ :-]?([0-9]{4}\.[0-9]{4,}(?:v[0-9]+)?)', readme_text, re.IGNORECASE)
    if not m:
        print("Error: arXiv ID pattern not found in README.md", file=sys.stderr)
        sys.exit(1)
    readme_id = m.group(1)

    cff_path = repo_root / "CITATION.cff"
    if not cff_path.is_file():
        print(f"Error: CITATION.cff not found: {cff_path}", file=sys.stderr)
        sys.exit(1)
    with open(cff_path, "r", encoding="utf-8") as fh:
        cff_data = yaml.safe_load(fh)

    preferred_citation = cff_data.get("preferred-citation", {})
    cff_id = preferred_citation.get("arxiv")
    if not cff_id:
         cff_url = preferred_citation.get("url", "")
         url_match = re.search(r'arxiv\.org/abs/([0-9]{4}\.[0-9]{4,}(?:v[0-9]+)?)', cff_url, re.IGNORECASE)
         if url_match:
             cff_id = url_match.group(1)
    if not cff_id:
        print("Error: 'arxiv' key or valid URL not found in CITATION.cff", file=sys.stderr)
        sys.exit(1)

    if readme_id.lower() != cff_id.lower():
        print(f"Error: arXiv ID mismatch! README: '{readme_id}' vs CITATION.cff: '{cff_id}'", file=sys.stderr)
        sys.exit(1)

    print(f"Success: README.md and CITATION.cff arXiv IDs match ('{readme_id}').")

except Exception as e:
    print(f"An unexpected error occurred: {e}", file=sys.stderr)
    sys.exit(1)