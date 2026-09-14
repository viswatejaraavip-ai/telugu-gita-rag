"""Three-way reranking comparison on the frozen held-out candidate lists.

Each row already holds the retriever's top-50, so this measures the only thing a
reranker can change: the ordering of those 50.
"""
import json, math, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ROWS = [json.loads(l) for l in (ROOT/"finetune"/"held_ce_eval.jsonl").open(encoding="utf-8") if l.strip()]
MAX_LEN = 192

def score(rank_fn, k=10):
    agg = defaultdict(lambda: {"n":0,"hit":0,"ndcg":0.0,"ranks":[]})
    tot = {"n":0,"hit":0,"ndcg":0.0,"rr":0.0}
    for r in ROWS:
        ranked = rank_fn(r); gold = r["positive_id"]
        rank = ranked.index(gold)+1 if gold in ranked else None
        nd  = 1.0/math.log2(rank+1) if (rank and rank<=k) else 0.0
        hit = 1 if (rank and rank<=k) else 0
        tot["n"]+=1; tot["hit"]+=hit; tot["ndcg"]+=nd; tot["rr"] += (1.0/rank) if rank else 0.0
        a=agg[r["kind"]]; a["n"]+=1; a["hit"]+=hit; a["ndcg"]+=nd
        if rank: a["ranks"].append(rank)
    return dict(recall=tot["hit"]/tot["n"], ndcg=tot["ndcg"]/tot["n"], mrr=tot["rr"]/tot["n"],
                per={k2:dict(n=v["n"], recall=v["hit"]/v["n"], ndcg=v["ndcg"]/v["n"],
                             median=sorted(v["ranks"])[len(v["ranks"])//2] if v["ranks"] else None)
                     for k2,v in agg.items()})

def ce_fn(path):
    from sentence_transformers import CrossEncoder
    m = CrossEncoder(path, max_length=MAX_LEN, device="cpu")
    def f(r):
        s = m.predict([(r["query"], c) for c in r["candidates"]],
                      batch_size=32, show_progress_bar=False)
        return [r["candidate_ids"][i] for i in np.argsort(-np.asarray(s))]
    return f

def show(tag, res):
    print(f"{tag:<26} recall@10={res['recall']:.3f}  NDCG@10={res['ndcg']:.3f}  MRR={res['mrr']:.3f}")
    for k,v in sorted(res["per"].items()):
        print(f"    {k:<9} n={v['n']:<4} recall@10={v['recall']:.3f}  NDCG@10={v['ndcg']:.3f}  median={v['median']}")

if __name__ == "__main__":
    ceiling = sum(1 for r in ROWS if r["positive_id"] in r["candidate_ids"])/len(ROWS)
    print(f"{len(ROWS)} held-out queries | 50 candidates each | ceiling recall@10 = {ceiling:.3f}\n")
    base = score(lambda r: r["candidate_ids"]); show("retriever (no rerank)", base); print()
    for tag, path in [("FINE-TUNED reranker", "/Users/mobilemy/Downloads/bge-reranker-gita-final"),
                      ("stock reranker",      "BAAI/bge-reranker-v2-m3")]:
        t=time.time(); res = score(ce_fn(path)); show(tag, res)
        d_r, d_n = res["recall"]-base["recall"], res["ndcg"]-base["ndcg"]
        gap = ceiling - base["recall"]
        print(f"    vs no-rerank: recall {d_r:+.3f}  NDCG {d_n:+.3f}   "
              f"captured {100*d_r/gap if gap else 0:.0f}% of headroom   [{time.time()-t:.0f}s]\n")
