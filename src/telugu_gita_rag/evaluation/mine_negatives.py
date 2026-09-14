"""Mine hard negatives from the CURRENT retriever.

A hard negative is a verse the retriever ranks highly for a query but which is
not the answer. Those are the confusions actually costing recall, so they are
what the model needs to learn to separate. Random negatives teach nothing.
"""
from __future__ import annotations
import json, pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from retrieval import (COLLECTION, _bm25, _client, bm25_tokens, embed,
                       load_corpus, _rrf)

ROOT = Path(__file__).resolve().parent.parent
N_NEG = 2          # hard negatives per query
SKIP_TOP = 1       # ignore rank 1..SKIP_TOP (often the gold itself)
POOL = 20

def main(which="train"):
    src = ROOT / "finetune" / f"{which}_queries.jsonl"
    vsrc = ROOT / "finetune" / f"{which}_verses.jsonl"
    dst = ROOT / "finetune" / f"{which}_triplets.jsonl"
    rows = [json.loads(l) for l in src.open(encoding="utf-8") if l.strip()]
    verses = {r["id"]: r for r in load_corpus()}
    # negatives must come from the same split, never leak held-out text into training
    allowed = {json.loads(l)["id"] for l in vsrc.open(encoding="utf-8") if l.strip()}
    print(f"{len(rows)} queries; negative pool restricted to {len(allowed)} {which} verses")

    qs = [r["query"] for r in rows]
    print("embedding queries...")
    QV = embed(qs, batch_size=8)

    cl, bm = _client(), _bm25()
    out, skipped = [], 0
    for r, qv in zip(rows, QV):
        hits = cl.query_points(COLLECTION, query=qv, limit=POOL, with_payload=True).points
        dense_ids = [h.payload["id"] for h in hits]
        sc = bm["model"].get_scores(bm25_tokens(r["query"]))
        order = sorted(range(len(sc)), key=lambda i: -sc[i])[:POOL]
        lex_ids = [bm["ids"][i] for i in order]
        ranked = [d for d, _ in sorted(_rrf([dense_ids, lex_ids]).items(),
                                       key=lambda kv: -kv[1])]
        negs = [d for d in ranked[SKIP_TOP:]
                if d != r["verse_id"] and d in allowed][:N_NEG]
        if not negs:
            skipped += 1; continue
        for nid in negs:
            out.append(dict(anchor=r["query"],
                            positive=verses[r["verse_id"]]["tatparyam_te"],
                            negative=verses[nid]["tatparyam_te"],
                            kind=r["kind"], positive_id=r["verse_id"], negative_id=nid))
    with dst.open("w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} triplets -> {dst.name}  ({skipped} queries had no negative)")
    from collections import Counter
    print("  by kind:", dict(Counter(o["kind"] for o in out)))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "train")
