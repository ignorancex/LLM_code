#!/usr/bin/env python3
"""
Unitig → genome coherence scorer (GFA uses 'a' lines)

Works with Python-3.6+

GFA a-line columns
------------------
 1  a
 2  unitig name
 3  offset in unitig        (ignored except for sorting)
 4  read_id:subcoord        (keep text before first ':')
 5  strand                  (+ always in your data)
 6  read length             (integer)

PAF columns used
----------------
 1  query id
 2  query length
 6  reference name
 8  reference start   (1-based in your files)

Output columns
--------------
 unitig  chr  n  k  R
    n  = reads in unitig
    k  = longest chain obeying the distance rule
    R  = k / n
"""

import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# ----------------------------------------------------------------------
# 1  parse GFA ('a' lines only)
# ----------------------------------------------------------------------


def parse_gfa_a(path: str) -> Dict[str, List[Tuple[str, int]]]:
    """
    Returns
    -------
    unitigs : {unitig → ordered list of (read id, read length)}
    """
    unitigs: Dict[str, List[Tuple[int, str, int]]] = defaultdict(list)  # (offset, read, length)

    with open(path, "rt") as fh:
        for ln in fh:
            if ln.startswith("a\t"):
                _, utg, offset, read_field, _strand, length = ln.rstrip("\n").split(
                    "\t"
                )
                read_id = read_field.split(":", 1)[0]
                off = int(offset)
                ln_int = int(length)
                unitigs[utg].append((off, read_id, ln_int))

    # sort reads by their offset along the unitig, keep read id and length
    ordered: Dict[str, List[Tuple[str, int]]] = {
        utg: [(r, l) for _o, r, l in sorted(lst, key=lambda x: x[0])]
        for utg, lst in unitigs.items()
    }
    return ordered


# ----------------------------------------------------------------------
# 2  parse PAF + length back-fill
# ----------------------------------------------------------------------


def parse_paf(
    path: str,
    wanted: set,
) -> Dict[str, List[Tuple[str, int]]]:
    maps: Dict[str, List[Tuple[str, int]]] = defaultdict(list)

    with open(path, "rt") as fh:
        for ln in fh:
            if ln[0] == "#":
                continue
            c = ln.rstrip("\n").split("\t")
            q = c[0]
            if q not in wanted:
                continue
            # qlen = int(c[1])
            chrom = c[5]
            ref_start = int(c[7])       # already 1-based in your file
            maps[q].append((chrom, ref_start))
    return maps


# ----------------------------------------------------------------------
# 3  longest distance-constrained chain (quadratic, adequate for bacteria)
# ----------------------------------------------------------------------


def best_chain(
    path: List[Tuple[str, int]],
    maps: Dict[str, List[Tuple[str, int]]],
) -> Tuple[int, str]:
    # build anchors: (index_in_unitig, chr, start, read_len)
    anchors: List[Tuple[int, str, int, int]] = []
    for idx, (r, ln) in enumerate(path, 1):
        for chr_, s in maps.get(r, []):
            anchors.append((idx, chr_, s, ln))

    if not anchors:
        return 0, ""

    anchors.sort(key=lambda x: (x[1], x[2]))          # by chr, start
    m = len(anchors)
    dp = [1] * m
    best_j = 0

    for j in range(1, m):
        idx_j, chr_j, s_j, _ln_j = anchors[j]
        best_val = 1
        for k in range(j):
            idx_k, chr_k, s_k, ln_k = anchors[k]
            if chr_k != chr_j or idx_k >= idx_j:
                continue
            if s_j < s_k or s_j >= s_k + ln_k + 50_000:
                continue
            cand = dp[k] + 1
            if cand > best_val:
                best_val = cand
        dp[j] = best_val
        if dp[j] > dp[best_j]:
            best_j = j

    return dp[best_j], anchors[best_j][1]


# ----------------------------------------------------------------------
# 4  CLI
# ----------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Unitig coherence (GFA 'a' lines).")
    p.add_argument("gfa", help="GFA file")
    p.add_argument("paf", help="PAF file")
    args = p.parse_args(argv)

    unitigs = parse_gfa_a(args.gfa)
    if not unitigs:
        import sys
        sys.exit("No 'a' lines found in the GFA; nothing to score.")

    wanted = {r for path in unitigs.values() for (r, _l) in path}
    maps = parse_paf(args.paf, wanted)
    
    all_n = 0
    all_k = 0

    print("unitig\tchr\tn\tk\tR")
    for utg, path in sorted(unitigs.items()):
        n = len(path)
        if n <= 2:            # skip single‑ and two-read unitigs
            continue
        k, chr_best = best_chain(path, maps)
        R = 0.0 if n == 0 else k / n
        print(f"{utg}\t{chr_best}\t{n}\t{k}\t{R:.3f}")
        all_n += n          # ← accumulate
        all_k += k          # ← accumulate

    # after the loop
    if all_n > 0:
        print(f"TOTAL\t-\t{all_n}\t{all_k}\t{all_k / all_n:.3f}")
    else:
        print("TOTAL\t-\t0\t0\t0.000")


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
