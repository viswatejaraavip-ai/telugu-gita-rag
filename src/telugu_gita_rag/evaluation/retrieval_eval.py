"""Retrieval eval: recall@k / MRR over a hand-labelled question->gold-verse set.

Run this BEFORE changing anything, so you have a baseline. Telugu RAG quality is
not assessable by reading a few answers.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieval import search

EVAL = Path(__file__).resolve().parent.parent / "eval" / "questions.jsonl"

def run(ks=(1, 3, 5, 10), pool=20):
    qs = [json.loads(l) for l in EVAL.open(encoding="utf-8") if l.strip()]
    by_kind = {}
    if not qs:
        print("no eval questions yet"); return
    hits = {k: 0 for k in ks}; rr = 0.0; misses = []
    for q in qs:
        got = [r["id"] for r in search(q["question_te"], top_k=pool)]
        gold = set(q["gold_ids"])
        rank = next((i + 1 for i, g in enumerate(got) if g in gold), None)
        if rank: rr += 1.0 / rank
        else: misses.append(q)
        for k in ks:
            if rank and rank <= k: hits[k] += 1
        by_kind.setdefault(q.get("kind","?"), []).append(rank)
    n = len(qs)
    print(f"n={n}  " + "  ".join(f"recall@{k}={hits[k]/n:.2f}" for k in ks)
          + f"  MRR={rr/n:.3f}")
    print()
    for kind, ranks in sorted(by_kind.items()):
        found = [r for r in ranks if r]
        print(f"  {kind:<9} n={len(ranks):<3} recall@10={sum(1 for r in found if r<=10)/len(ranks):.2f}"
              f"  median_rank={sorted(found)[len(found)//2] if found else '-'}")
    if misses:
        print(f"\n{len(misses)} misses (not in top {pool}):")
        for m in misses[:10]:
            print(f"  {m['gold_ids']}  {m['question_te'][:60]}")

if __name__ == "__main__":
    run()
