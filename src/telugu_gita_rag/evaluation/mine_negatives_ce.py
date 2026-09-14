"""Mine hard negatives for CROSS-ENCODER training.

Two differences from the bi-encoder miner:
  1. More negatives per query. A cross-encoder has no in-batch negatives - every
     negative must be supplied explicitly, because scoring a pair costs a full
     forward pass. So we gather N_NEG per query instead of 2.
  2. Grouped output: one row per query holding a list of negatives, which is the
     shape listwise ranking losses expect.

Mined with the FINE-TUNED retriever, so these are the confusions the *current*
system makes - round-two negatives, harder than round one.
"""
from __future__ import annotations
import json, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from retrieval import (MODEL_NAME, _bm25, _rrf, bm25_tokens, embed, load_corpus,
                       retrieval_text)

ROOT = Path(__file__).resolve().parent.parent
N_NEG = 8
POOL  = 50

def main(which="train"):
    src  = ROOT / "finetune" / f"{which}_queries.jsonl"
    vsrc = ROOT / "finetune" / f"{which}_verses.jsonl"
    dst  = ROOT / "finetune" / f"{which}_ce.jsonl"

    recs   = load_corpus()
    by_id  = {r["id"]: r for r in recs}
    ids    = [r["id"] for r in recs]
    allowed = {json.loads(l)["id"] for l in vsrc.open(encoding="utf-8") if l.strip()}

    rows = [json.loads(l) for l in src.open(encoding="utf-8") if l.strip()]
    print(f"retriever: {MODEL_NAME}")
    print(f"{len(rows)} queries; negatives from {len(allowed)} {which} verses; N_NEG={N_NEG}")

    print("embedding corpus...")
    V = np.asarray(embed([retrieval_text(r) for r in recs], batch_size=8))
    print("embedding queries...")
    Q = np.asarray(embed([r["query"] for r in rows], batch_size=8))

    bm = _bm25()
    out, nneg = [], []
    for r, qv in zip(rows, Q):
        sims = V @ qv
        dense_ids = [ids[i] for i in np.argsort(-sims)[:POOL]]
        sc = bm["model"].get_scores(bm25_tokens(r["query"]))
        lex_ids = [bm["ids"][i] for i in sorted(range(len(sc)), key=lambda i: -sc[i])[:POOL]]
        ranked = [d for d, _ in sorted(_rrf([dense_ids, lex_ids]).items(), key=lambda kv: -kv[1])]
        negs = [d for d in ranked if d != r["verse_id"] and d in allowed][:N_NEG]
        if len(negs) < 2: continue
        nneg.append(len(negs))
        out.append(dict(query=r["query"], kind=r["kind"],
                        positive=by_id[r["verse_id"]]["tatparyam_te"],
                        positive_id=r["verse_id"],
                        negatives=[by_id[n]["tatparyam_te"] for n in negs],
                        negative_ids=negs))
    with dst.open("w", encoding="utf-8") as f:
        for o in out: f.write(json.dumps(o, ensure_ascii=False) + "\n")
    pairs = sum(1 + len(o["negatives"]) for o in out)
    print(f"wrote {len(out)} grouped rows -> {dst.name}")
    print(f"  mean negatives/query : {np.mean(nneg):.1f}")
    print(f"  total (q,doc) pairs  : {pairs:,}   <- each is one forward pass per epoch")
    from collections import Counter
    print("  by kind:", dict(Counter(o["kind"] for o in out)))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "train")
