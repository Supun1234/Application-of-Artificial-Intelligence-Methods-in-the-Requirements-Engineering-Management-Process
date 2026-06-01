"""
inter_annotator_agreement.py
=============================
Computes inter-annotator agreement (Cohen's Kappa + raw accuracy)
between three annotators (Author, ChatGPT, Claude) across all three
RE pipeline stages.

Usage
-----
  python inter_annotator_agreement.py \
      --author   annotations_author.json \
      --chatgpt  annotations_chatgpt.json \
      --claude   annotations_claude.json \
      --output   iaa_results.json

Expected JSON format for each annotation file
----------------------------------------------
[
  {
    "id": "BANK_01",
    "complete": true,           <- Stage 1: true = complete, false = incomplete
    "type": "FR",               <- Stage 2: "FR" or "NFR"
    "links": ["BANK_02"]        <- Stage 3: list of dependency IDs (can be [])
  },
  ...
]

Output
------
- Prints a full report to stdout
- Saves iaa_results.json with all numbers thesis-ready
"""

import argparse
import json
import itertools
from collections import defaultdict
from sklearn.metrics import cohen_kappa_score, accuracy_score


# ─────────────────────────────────────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def load(path: str) -> dict:
    """Load a JSON annotation file. Returns dict keyed by requirement ID."""
    with open(path) as f:
        data = json.load(f)
    result = {}
    for r in data:
        result[r["id"]] = {
        "complete": r["truth"]["complete"],
        "type":     r["truth"]["type"],
        "links":    r.get("links", []),
    }
    return result


def align(a: dict, b: dict) -> list:
    """Return list of IDs present in both annotation sets (intersection)."""
    common = sorted(set(a.keys()) & set(b.keys()))
    if len(common) < len(a) or len(common) < len(b):
        only_a = set(a.keys()) - set(b.keys())
        only_b = set(b.keys()) - set(a.keys())
        if only_a:
            print(f"  ⚠️  IDs only in first file (skipped): {sorted(only_a)}")
        if only_b:
            print(f"  ⚠️  IDs only in second file (skipped): {sorted(only_b)}")
    return common


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — completeness (binary)
# ─────────────────────────────────────────────────────────────────────────────

def s1_vectors(ann: dict, ids: list) -> list:
    """Extract Stage 1 labels as 1 (complete) / 0 (incomplete)."""
    return [1 if ann[i]["complete"] else 0 for i in ids]


def stage1_agreement(a: dict, b: dict, name_a: str, name_b: str) -> dict:
    ids = align(a, b)
    va, vb = s1_vectors(a, ids), s1_vectors(b, ids)
    kappa = cohen_kappa_score(va, vb)
    acc   = accuracy_score(va, vb)
    agree = sum(x == y for x, y in zip(va, vb))

    # per-ID breakdown
    mismatches = [ids[i] for i in range(len(ids)) if va[i] != vb[i]]

    return {
        "pair":       f"{name_a} vs {name_b}",
        "n":          len(ids),
        "agreed":     agree,
        "kappa":      round(kappa, 4),
        "accuracy":   round(acc,   4),
        "mismatches": mismatches,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — FR/NFR classification (binary)
# ─────────────────────────────────────────────────────────────────────────────

def s2_vectors(ann: dict, ids: list) -> list:
    return [ann[i]["type"] for i in ids]


def stage2_agreement(a: dict, b: dict, name_a: str, name_b: str) -> dict:
    ids = align(a, b)
    va, vb = s2_vectors(a, ids), s2_vectors(b, ids)
    kappa = cohen_kappa_score(va, vb)
    acc   = accuracy_score(va, vb)
    agree = sum(x == y for x, y in zip(va, vb))
    mismatches = [
        {"id": ids[i], name_a: va[i], name_b: vb[i]}
        for i in range(len(ids)) if va[i] != vb[i]
    ]
    return {
        "pair":       f"{name_a} vs {name_b}",
        "n":          len(ids),
        "agreed":     agree,
        "kappa":      round(kappa, 4),
        "accuracy":   round(acc,   4),
        "mismatches": mismatches,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — dependency links (set-based agreement)
# ─────────────────────────────────────────────────────────────────────────────

def links_to_pairs(ann: dict, ids: list) -> set:
    """Convert link lists to a set of directed (src, tgt) pairs."""
    pairs = set()
    for i in ids:
        for tgt in ann[i].get("links", []):
            if tgt in ann:          # only count links within the shared set
                pairs.add((i, tgt))
    return pairs


def stage3_agreement(a: dict, b: dict, name_a: str, name_b: str) -> dict:
    ids = align(a, b)
    pa  = links_to_pairs(a, ids)
    pb  = links_to_pairs(b, ids)

    union     = pa | pb
    intersect = pa & pb
    only_a    = pa - pb
    only_b    = pb - pa

    # Jaccard similarity
    jaccard = len(intersect) / len(union) if union else 1.0

    # Convert to binary vectors over the union for Kappa
    all_pairs = sorted(union)
    va = [1 if p in pa else 0 for p in all_pairs]
    vb = [1 if p in pb else 0 for p in all_pairs]

    if len(set(va)) < 2 or len(set(vb)) < 2:
        # degenerate case — one annotator marked nothing
        kappa = 0.0
    else:
        kappa = cohen_kappa_score(va, vb)

    return {
        "pair":          f"{name_a} vs {name_b}",
        "n_requirements": len(ids),
        "links_a":       len(pa),
        "links_b":       len(pb),
        "agreed_links":  len(intersect),
        "jaccard":       round(jaccard, 4),
        "kappa":         round(kappa,   4),
        "only_in_a":     sorted(only_a),
        "only_in_b":     sorted(only_b),
    }


# ─────────────────────────────────────────────────────────────────────────────
# FLEISS-STYLE MULTI-ANNOTATOR KAPPA (3-way)
# ─────────────────────────────────────────────────────────────────────────────

def majority_vote(labels: list) -> str:
    """Return the most common label; tie goes to first."""
    from collections import Counter
    return Counter(labels).most_common(1)[0][0]


def three_way_agreement(anns: dict, ids: list, field: str,
                        binary: bool = True) -> dict:
    """
    For S1/S2: compute proportion of items where all 3 agree,
    at-least-2 agree, and majority label.
    """
    all3   = 0
    at2    = 0
    for i in ids:
        vals = [anns[name][i][field] for name in anns if i in anns[name]]
        if len(set(vals)) == 1:
            all3 += 1
            at2  += 1
        elif len(vals) >= 2:
            # at least 2 same
            from collections import Counter
            if Counter(vals).most_common(1)[0][1] >= 2:
                at2 += 1
    return {
        "n":                len(ids),
        "all_three_agree":  all3,
        "at_least_two":     at2,
        "pct_all_agree":    round(all3 / len(ids) * 100, 1) if ids else 0,
        "pct_two_agree":    round(at2  / len(ids) * 100, 1) if ids else 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# KAPPA INTERPRETATION
# ─────────────────────────────────────────────────────────────────────────────

def interpret_kappa(k: float) -> str:
    if k < 0:     return "Poor (worse than chance)"
    if k < 0.20:  return "Slight"
    if k < 0.40:  return "Fair"
    if k < 0.60:  return "Moderate"
    if k < 0.80:  return "Substantial"
    return "Almost perfect"


# ─────────────────────────────────────────────────────────────────────────────
# REPORT PRINTER
# ─────────────────────────────────────────────────────────────────────────────

def print_report(s1: list, s2: list, s3: list,
                 s1_3way: dict, s2_3way: dict):

    W = 62
    print("\n" + "=" * W)
    print("  INTER-ANNOTATOR AGREEMENT REPORT")
    print("  Annotators: Author · ChatGPT · Claude")
    print("=" * W)

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    print("\n1️⃣  Stage 1 — Completeness (complete / incomplete)")
    print(f"   {'Pair':<28} {'κ':>6}  {'Interp.':<22}  {'Agree':>6}")
    print(f"   {'-'*28} {'-'*6}  {'-'*22}  {'-'*6}")
    for r in s1:
        interp = interpret_kappa(r["kappa"])
        pct    = f"{r['agreed']}/{r['n']}"
        print(f"   {r['pair']:<28} {r['kappa']:>6.3f}  {interp:<22}  {pct:>6}")

    print(f"\n   3-way summary:")
    print(f"   All three agree : {s1_3way['all_three_agree']}/{s1_3way['n']}  "
          f"({s1_3way['pct_all_agree']}%)")
    print(f"   At least 2 agree: {s1_3way['at_least_two']}/{s1_3way['n']}  "
          f"({s1_3way['pct_two_agree']}%)")

    for r in s1:
        if r["mismatches"]:
            print(f"\n   Mismatches — {r['pair']}: {r['mismatches']}")

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    print("\n2️⃣  Stage 2 — Classification (FR / NFR)")
    print(f"   {'Pair':<28} {'κ':>6}  {'Interp.':<22}  {'Agree':>6}")
    print(f"   {'-'*28} {'-'*6}  {'-'*22}  {'-'*6}")
    for r in s2:
        interp = interpret_kappa(r["kappa"])
        pct    = f"{r['agreed']}/{r['n']}"
        print(f"   {r['pair']:<28} {r['kappa']:>6.3f}  {interp:<22}  {pct:>6}")

    print(f"\n   3-way summary:")
    print(f"   All three agree : {s2_3way['all_three_agree']}/{s2_3way['n']}  "
          f"({s2_3way['pct_all_agree']}%)")
    print(f"   At least 2 agree: {s2_3way['at_least_two']}/{s2_3way['n']}  "
          f"({s2_3way['pct_two_agree']}%)")

    for r in s2:
        if r["mismatches"]:
            print(f"\n   Mismatches — {r['pair']}:")
            for m in r["mismatches"]:
                print(f"     {m['id']}: {list(m.values())[1]} vs {list(m.values())[2]}")

    # ── Stage 3 ──────────────────────────────────────────────────────────────
    print("\n3️⃣  Stage 3 — Dependency Links (Jaccard + κ)")
    print(f"   {'Pair':<28} {'Jaccard':>8}  {'κ':>6}  {'Interp.'}")
    print(f"   {'-'*28} {'-'*8}  {'-'*6}  {'-'*20}")
    for r in s3:
        interp = interpret_kappa(r["kappa"])
        print(f"   {r['pair']:<28} {r['jaccard']:>8.3f}  "
              f"{r['kappa']:>6.3f}  {interp}")
        print(f"      links: A={r['links_a']}, B={r['links_b']}, "
              f"shared={r['agreed_links']}")
        if r["only_in_a"]:
            print(f"      Only in first : {r['only_in_a'][:5]}"
                  f"{'...' if len(r['only_in_a'])>5 else ''}")
        if r["only_in_b"]:
            print(f"      Only in second: {r['only_in_b'][:5]}"
                  f"{'...' if len(r['only_in_b'])>5 else ''}")

    # ── Thesis paragraph ──────────────────────────────────────────────────────
    avg_s1_kappa = sum(r["kappa"] for r in s1) / len(s1)
    avg_s2_kappa = sum(r["kappa"] for r in s2) / len(s2)
    avg_s3_jacc  = sum(r["jaccard"] for r in s3) / len(s3)

    print(f"\n{'─'*W}")
    print("  SUGGESTED THESIS TEXT (Section 5.2 / 5.7)")
    print(f"{'─'*W}")
    print(f"""
  To mitigate the absence of a pre-existing multi-stage benchmark,
  the 30-requirement evaluation dataset was independently annotated
  by three raters: the thesis author, ChatGPT-4o, and Claude Sonnet.
  Each annotator labeled all three dimensions — completeness (Stage 1),
  functional type (Stage 2), and dependency links (Stage 3) — without
  access to the pipeline's outputs.

  Inter-annotator agreement was measured using Cohen's Kappa (κ) for
  Stage 1 and Stage 2, and Jaccard similarity for Stage 3 dependency
  links, where set-based overlap is the appropriate measure for
  multi-label link annotation.

  Stage 1 pairwise kappa ranged from {min(r['kappa'] for r in s1):.3f} to
  {max(r['kappa'] for r in s1):.3f} (mean κ={avg_s1_kappa:.3f},
  '{interpret_kappa(avg_s1_kappa)}' agreement), indicating that
  completeness assessment is a well-defined, reliably judgeable task.

  Stage 2 pairwise kappa ranged from {min(r['kappa'] for r in s2):.3f} to
  {max(r['kappa'] for r in s2):.3f} (mean κ={avg_s2_kappa:.3f},
  '{interpret_kappa(avg_s2_kappa)}' agreement), consistent with prior
  work on FR/NFR classification annotation difficulty.

  Stage 3 Jaccard similarity averaged {avg_s3_jacc:.3f} across annotator
  pairs, reflecting the inherent subjectivity of dependency identification
  in natural language requirements — a known challenge in the RE
  traceability literature (Cleland-Huang et al., 2014).
""")
    print("=" * W)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Inter-annotator agreement for RE pipeline annotations")
    parser.add_argument("--author",  required=True,
                        help="Author annotation JSON")
    parser.add_argument("--chatgpt", required=True,
                        help="ChatGPT annotation JSON")
    parser.add_argument("--claude",  required=True,
                        help="Claude annotation JSON")
    parser.add_argument("--output",  default="iaa_results.json",
                        help="Output JSON path")
    args = parser.parse_args()

    print(f"[IAA] Loading annotations…")
    anns = {
        "Author":  load(args.author),
        "ChatGPT": load(args.chatgpt),
        "Claude":  load(args.claude),
    }

    # shared IDs across all three
    all_ids = sorted(
        set(anns["Author"].keys()) &
        set(anns["ChatGPT"].keys()) &
        set(anns["Claude"].keys())
    )
    print(f"[IAA] {len(all_ids)} requirements common to all three annotators.")

    pairs = list(itertools.combinations(anns.keys(), 2))

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    print("\n[IAA] Computing Stage 1 (completeness)…")
    s1_results = [
        stage1_agreement(anns[a], anns[b], a, b)
        for a, b in pairs
    ]
    s1_3way = three_way_agreement(anns, all_ids, "complete")

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    print("[IAA] Computing Stage 2 (FR/NFR)…")
    s2_results = [
        stage2_agreement(anns[a], anns[b], a, b)
        for a, b in pairs
    ]
    s2_3way = three_way_agreement(anns, all_ids, "type")

    # ── Stage 3 ──────────────────────────────────────────────────────────────
    print("[IAA] Computing Stage 3 (dependency links)…")
    s3_results = [
        stage3_agreement(anns[a], anns[b], a, b)
        for a, b in pairs
    ]

    # ── Print full report ─────────────────────────────────────────────────────
    print_report(s1_results, s2_results, s3_results, s1_3way, s2_3way)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output = {
        "n_requirements": len(all_ids),
        "annotators":     list(anns.keys()),
        "stage1": {"pairs": s1_results, "three_way": s1_3way},
        "stage2": {"pairs": s2_results, "three_way": s2_3way},
        "stage3": {"pairs": s3_results},
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 Results saved to: {args.output}")


if __name__ == "__main__":
    main()
