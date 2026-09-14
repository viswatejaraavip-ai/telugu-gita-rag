"""Freeze the retriever's top-50 candidates for the held-out queries.

A reranker is only ever asked to reorder what retrieval returned, so the eval
set must BE that output. We ship the candidate lists (and the retriever's own
ordering, as the no-rerank baseline) so Colab needs no index.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from retrieval import (MODEL_NAME, _bm25, _rrf, bm25_tokens, embed, load_corpus,
                       retrieval_text)

ROOT = Path(__file__).resolve().parent.parent
POOL = 50

def main():
    recs  = load_corpus(); ids = [r["id"] for r in recs]
    by_id = {r["id"]: r["tatparyam_te"] for r in recs}
    rows  = [json.loads(l) for l in (ROOT/"finetune"/"held_queries.jsonl").open(encoding="utf-8") if l.strip()]
    print(f"retriever: {MODEL_NAME}  |  {len(rows)} held-out queries")
    V = np.asarray(embed([retrieval_text(r) for r in recs], batch_size=8))
    Q = np.asarray(embed([r["query"] for r in rows], batch_size=8))
    bm = _bm25(); out = []; has_gold = 0
    for r, qv in zip(rows, Q):
        sims = V @ qv
        dense_ids = [ids[i] for i in np.argsort(-sims)[:POOL]]
        sc = bm["model"].get_scores(bm25_tokens(r["query"]))
        lex_ids = [bm["ids"][i] for i in sorted(range(len(sc)), key=lambda i: -sc[i])[:POOL]]
        ranked = [d for d,_ in sorted(_rrf([dense_ids, lex_ids]).items(), key=lambda kv:-kv[1])][:POOL]
        if r["verse_id"] in ranked: has_gold += 1
        out.append(dict(query=r["query"], kind=r["kind"], positive_id=r["verse_id"],
                        candidate_ids=ranked, candidates=[by_id[c] for c in ranked]))
    dst = ROOT/"finetune"/"held_ce_eval.jsonl"
    with dst.open("w", encoding="utf-8") as f:
        for o in out: f.write(json.dumps(o, ensure_ascii=False)+"\n")
    print(f"wrote {len(out)} rows -> {dst.name}")
    print(f"  gold present in top-{POOL}: {has_gold}/{len(out)} = {has_gold/len(out):.3f}  <- reranker ceiling")

if __name__ == "__main__":
    main()
