"""
merge_annotations.py
=====================
Merges three annotator JSON files into a single gold-standard ground truth
using majority vote across all three stages.

Usage
-----
  python merge_annotations.py \
      --author   annotations_author.json \
      --chatgpt  annotations_chatgpt.json \
      --claude   annotations_claude.json \
      --output   ground_truth_merged.json

Output format
-------------
Same structure as extended_banking_batch_v2.json — drop-in replacement.
The "truth" field reflects the majority vote, not any single annotator.

[
  {
    "id": "BANK_01",
    "text": "...",                  <- copied from author file
    "truth": {
      "complete": true,            <- majority vote (2 or 3 of 3 agree)
      "type": "FR"                 <- majority vote
    },
    "links": ["BANK_02"],          <- union of links agreed by >= 2 annotators
    "merge_meta": {
      "s1_votes": {"Author": true, "ChatGPT": true, "Claude": true},
      "s1_agreement": "unanimous",
      "s2_votes": {"Author": "FR", "ChatGPT": "FR", "Claude": "FR"},
      "s2_agreement": "unanimous",
      "s3_links_author":  ["BANK_02"],
      "s3_links_chatgpt": ["BANK_02"],
      "s3_links_claude":  ["BANK_02"],
      "s3_agreed_links":  ["BANK_02"],
      "s3_disputed_links": []
    }
  },
  ...
]
"""

import argparse
import json
from collections import Counter


# ─────────────────────────────────────────────────────────────────────────────
# LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    # support both formats:
    #   {"id":..., "complete":..., "type":..., "links":[...]}  <- annotator format
    #   {"id":..., "truth":{"complete":..., "type":...}, ...}  <- dataset format
    result = {}
    for r in data:
        if "truth" in r:
            entry = {
                "id":       r["id"],
                "text":     r.get("text", ""),
                "complete": r["truth"]["complete"],
                "type":     r["truth"]["type"],
                "links":    r.get("links", []),
            }
        else:
            entry = {
                "id":       r["id"],
                "text":     r.get("text", ""),
                "complete": r["complete"],
                "type":     r["type"],
                "links":    r.get("links", []),
            }
        result[r["id"]] = entry
    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAJORITY VOTE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def majority_bool(votes: list[bool]) -> tuple[bool, str]:
    """
    Majority vote for a boolean field (Stage 1).
    Returns (winner, agreement_label).
    """
    true_count  = sum(1 for v in votes if v is True)
    false_count = len(votes) - true_count

    if true_count == len(votes):
        return True, "unanimous"
    if false_count == len(votes):
        return False, "unanimous"
    if true_count > false_count:
        return True, "majority"
    return False, "majority"


def majority_str(votes: list[str]) -> tuple[str, str]:
    """
    Majority vote for a string field (Stage 2).
    Returns (winner, agreement_label).
    """
    counts = Counter(votes)
    winner, top_count = counts.most_common(1)[0]

    if top_count == len(votes):
        return winner, "unanimous"
    if top_count > 1:
        return winner, "majority"
    # 3-way tie (all different) — fall back to first annotator
    return votes[0], "disputed"


def majority_links(links_per_annotator: list[list[str]],
                   threshold: int = 2) -> tuple[list[str], list[str]]:
    """
    For each candidate link, count how many annotators included it.
    Include in gold standard if >= threshold annotators agree.

    Returns (agreed_links, disputed_links).
    """
    link_counts: dict[str, int] = Counter()
    for links in links_per_annotator:
        for link in links:
            link_counts[link] += 1

    agreed   = sorted(l for l, c in link_counts.items() if c >= threshold)
    disputed = sorted(l for l, c in link_counts.items() if c < threshold)
    return agreed, disputed


# ─────────────────────────────────────────────────────────────────────────────
# MERGE
# ─────────────────────────────────────────────────────────────────────────────

def merge(author: dict, chatgpt: dict, claude: dict,
          link_threshold: int = 2) -> list[dict]:
    """
    Merge three annotation dicts into a single gold-standard list.
    Uses the author file as the source of requirement text and ordering.
    """
    # IDs present in all three
    common_ids = sorted(
        set(author.keys()) & set(chatgpt.keys()) & set(claude.keys())
    )

    only_author  = set(author.keys())  - set(chatgpt.keys()) - set(claude.keys())
    only_chatgpt = set(chatgpt.keys()) - set(author.keys())  - set(claude.keys())
    only_claude  = set(claude.keys())  - set(author.keys())  - set(chatgpt.keys())

    if only_author:
        print(f"⚠️  IDs only in author file (skipped): {sorted(only_author)}")
    if only_chatgpt:
        print(f"⚠️  IDs only in ChatGPT file (skipped): {sorted(only_chatgpt)}")
    if only_claude:
        print(f"⚠️  IDs only in Claude file (skipped): {sorted(only_claude)}")

    merged = []
    stats  = {"unanimous_s1": 0, "majority_s1": 0,
               "unanimous_s2": 0, "majority_s2": 0, "disputed_s2": 0,
               "total_agreed_links": 0, "total_disputed_links": 0}

    for rid in common_ids:
        a = author[rid]
        g = chatgpt[rid]
        c = claude[rid]

        # ── Stage 1: completeness ─────────────────────────────────────────
        s1_votes = [a["complete"], g["complete"], c["complete"]]
        s1_winner, s1_agreement = majority_bool(s1_votes)
        stats[f"{s1_agreement}_s1"] = stats.get(f"{s1_agreement}_s1", 0) + 1

        # ── Stage 2: FR/NFR ───────────────────────────────────────────────
        s2_votes = [a["type"], g["type"], c["type"]]
        s2_winner, s2_agreement = majority_str(s2_votes)
        stats[f"{s2_agreement}_s2"] = stats.get(f"{s2_agreement}_s2", 0) + 1

        # ── Stage 3: links ────────────────────────────────────────────────
        all_links = [a.get("links", []), g.get("links", []), c.get("links", [])]
        agreed_links, disputed_links = majority_links(all_links, link_threshold)
        stats["total_agreed_links"]   += len(agreed_links)
        stats["total_disputed_links"] += len(disputed_links)

        merged.append({
            "id":   rid,
            "text": a.get("text", ""),
            "truth": {
                "complete": s1_winner,
                "type":     s2_winner,
            },
            "links": agreed_links,
            "merge_meta": {
                "s1_votes": {
                    "Author":  a["complete"],
                    "ChatGPT": g["complete"],
                    "Claude":  c["complete"],
                },
                "s1_agreement": s1_agreement,
                "s2_votes": {
                    "Author":  a["type"],
                    "ChatGPT": g["type"],
                    "Claude":  c["type"],
                },
                "s2_agreement": s2_agreement,
                "s3_links_author":   sorted(a.get("links", [])),
                "s3_links_chatgpt":  sorted(g.get("links", [])),
                "s3_links_claude":   sorted(c.get("links", [])),
                "s3_agreed_links":   agreed_links,
                "s3_disputed_links": disputed_links,
            },
        })

    return merged, stats, common_ids


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_report(merged: list, stats: dict, common_ids: list):
    n = len(merged)
    print(f"\n{'='*56}")
    print(f"  MERGE REPORT — {n} requirements")
    print(f"{'='*56}")

    # ── Stage 1 ──────────────────────────────────────────────────────────
    complete_count   = sum(1 for r in merged if r["truth"]["complete"])
    incomplete_count = n - complete_count
    unanimous_s1     = sum(1 for r in merged
                           if r["merge_meta"]["s1_agreement"] == "unanimous")
    majority_s1      = n - unanimous_s1

    print(f"\n Stage 1 — Completeness")
    print(f"   Complete (gold)   : {complete_count}")
    print(f"   Incomplete (gold) : {incomplete_count}")
    print(f"   Unanimous         : {unanimous_s1}/{n} ({unanimous_s1/n*100:.0f}%)")
    print(f"   Majority only     : {majority_s1}/{n} ({majority_s1/n*100:.0f}%)")

    disputed_s1 = [r["id"] for r in merged
                   if len(set(r["merge_meta"]["s1_votes"].values())) == 3]
    if disputed_s1:
        print(f"   ⚠️  3-way split (author wins): {disputed_s1}")

    # ── Stage 2 ──────────────────────────────────────────────────────────
    fr_count  = sum(1 for r in merged if r["truth"]["type"] == "FR")
    nfr_count = n - fr_count
    unanimous_s2 = sum(1 for r in merged
                       if r["merge_meta"]["s2_agreement"] == "unanimous")
    majority_s2  = sum(1 for r in merged
                       if r["merge_meta"]["s2_agreement"] == "majority")
    disputed_s2  = sum(1 for r in merged
                       if r["merge_meta"]["s2_agreement"] == "disputed")

    print(f"\n Stage 2 — FR/NFR Classification")
    print(f"   FR (gold)    : {fr_count}")
    print(f"   NFR (gold)   : {nfr_count}")
    print(f"   Unanimous    : {unanimous_s2}/{n} ({unanimous_s2/n*100:.0f}%)")
    print(f"   Majority     : {majority_s2}/{n} ({majority_s2/n*100:.0f}%)")
    if disputed_s2 > 0:
        print(f"   ⚠️  3-way split (author wins): {disputed_s2}")
        for r in merged:
            if r["merge_meta"]["s2_agreement"] == "disputed":
                v = r["merge_meta"]["s2_votes"]
                print(f"      {r['id']}: "
                      f"Author={v['Author']} ChatGPT={v['ChatGPT']} "
                      f"Claude={v['Claude']} → gold={r['truth']['type']}")

    # ── Stage 3 ──────────────────────────────────────────────────────────
    print(f"\n Stage 3 — Dependency Links")
    print(f"   Gold links (≥2 agree)  : {stats['total_agreed_links']}")
    print(f"   Disputed links (1/3)   : {stats['total_disputed_links']}")

    # requirements where annotators had very different link counts
    big_gaps = []
    for r in merged:
        m = r["merge_meta"]
        counts = [len(m["s3_links_author"]),
                  len(m["s3_links_chatgpt"]),
                  len(m["s3_links_claude"])]
        if max(counts) - min(counts) >= 3:
            big_gaps.append((r["id"], counts))
    if big_gaps:
        print(f"   ⚠️  Large disagreements (author/gpt/claude link counts):")
        for rid, counts in big_gaps:
            print(f"      {rid}: {counts[0]} / {counts[1]} / {counts[2]} links")

    # ── Full detail table ─────────────────────────────────────────────────
    print(f"\n {'ID':<10} {'S1-gold':>8} {'S1-agree':>10}  "
          f"{'S2-gold':>7} {'S2-agree':>10}  {'Links':>5}")
    print(f" {'-'*10} {'-'*8} {'-'*10}  {'-'*7} {'-'*10}  {'-'*5}")
    for r in merged:
        m   = r["merge_meta"]
        s1g = "COMP" if r["truth"]["complete"] else "INCOMP"
        s2g = r["truth"]["type"]
        print(f" {r['id']:<10} {s1g:>8} {m['s1_agreement']:>10}  "
              f"{s2g:>7} {m['s2_agreement']:>10}  "
              f"{len(r['links']):>5}")

    print(f"\n{'='*56}")
    print(f"  Gold standard saved. Use in your notebook:")
    print(f"    with open('ground_truth_merged.json') as f:")
    print(f"        batch = json.load(f)")
    print(f"{'='*56}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Merge three annotator files into one gold-standard JSON")
    parser.add_argument("--author",    required=True,
                        help="Author annotation JSON")
    parser.add_argument("--chatgpt",   required=True,
                        help="ChatGPT annotation JSON")
    parser.add_argument("--claude",    required=True,
                        help="Claude annotation JSON")
    parser.add_argument("--output",    default="ground_truth_merged.json",
                        help="Output path (default: ground_truth_merged.json)")
    parser.add_argument("--threshold", type=int, default=2,
                        help="Min annotators that must agree on a link "
                             "for it to be included (default: 2)")
    args = parser.parse_args()

    print(f"[Merge] Loading annotation files…")
    author  = load(args.author)
    chatgpt = load(args.chatgpt)
    claude  = load(args.claude)

    print(f"[Merge] Author: {len(author)} reqs | "
          f"ChatGPT: {len(chatgpt)} reqs | "
          f"Claude: {len(claude)} reqs")

    merged, stats, common_ids = merge(
        author, chatgpt, claude, args.threshold)

    with open(args.output, "w") as f:
        json.dump(merged, f, indent=2)

    print_report(merged, stats, common_ids)
    print(f"💾  Saved to: {args.output}")


if __name__ == "__main__":
    main()
