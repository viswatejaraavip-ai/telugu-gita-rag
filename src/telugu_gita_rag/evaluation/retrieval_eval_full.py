"""Full retrieval eval: recall@k, NDCG@10, MRR — for any model, on any query set.

Dense scores are computed with numpy over the 699-verse corpus (it is tiny), so
swapping models needs no Qdrant rebuild. BM25 is model-independent and reused.
Corpus embeddings are cached per model.
"""
from __future__ import annotations
import argparse, hashlib, json, math, random, sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from retrieval import _bm25, _rrf, bm25_tokens, load_corpus, retrieval_text

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "index" / "embcache"; CACHE.mkdir(parents=True, exist_ok=True)

def corpus_vectors(model, model_key, texts):
    f = CACHE / f"corpus-{hashlib.md5(model_key.encode()).hexdigest()[:10]}.npy"
    if f.exists():
        v = np.load(f)
        if v.shape[0] == len(texts): return v
    v = np.asarray(model.encode(texts, batch_size=8, normalize_embeddings=True,
                                show_progress_bar=True))
    np.save(f, v); return v

def ndcg_at_k(ranked, gold, k=10):
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in gold)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal if ideal else 0.0

_ce = None
def reranker(name="BAAI/bge-reranker-v2-m3"):
    global _ce
    if _ce is None:
        from sentence_transformers import CrossEncoder
        _ce = CrossEncoder(name, max_length=512, device="cpu")
    return _ce

def run(model_path, qfile, n, seed, pool=50, rerank=False, rerank_model="BAAI/bge-reranker-v2-m3"):
    from sentence_transformers import SentenceTransformer
    recs = load_corpus()
    ids = [r["id"] for r in recs]
    texts = [retrieval_text(r) for r in recs]

    model = SentenceTransformer(model_path, device="cpu"); model.max_seq_length = 512
    V = corpus_vectors(model, model_path, texts)

    rows = [json.loads(l) for l in open(qfile, encoding="utf-8") if l.strip()]
    for r in rows:
        r.setdefault("gold_ids", [r["verse_id"]] if "verse_id" in r else [])
        r.setdefault("query", r.get("question_te", ""))
    if n and n < len(rows):
        # stratified by kind so the mix matches the full set
        byk = defaultdict(list)
        for r in rows: byk[r.get("kind", "?")].append(r)
        rnd = random.Random(seed); out = []
        for k, v in byk.items():
            rnd.shuffle(v)
            out += v[:max(1, round(n * len(v) / len(rows)))]
        rows = out[:n]

    Q = np.asarray(model.encode([r["query"] for r in rows], batch_size=8,
                                normalize_embeddings=True, show_progress_bar=True))
    bm = _bm25()
    by_id = {rec["id"]: rec["tatparyam_te"] for rec in recs}
    ks = (1, 3, 5, 10, 20, 50)
    hit = {k: 0 for k in ks}; rr = 0.0; ndcg = 0.0
    per = defaultdict(lambda: {"n": 0, "hit10": 0, "ndcg": 0.0, "ranks": []})
    for r, qv in zip(rows, Q):
        sims = V @ qv
        dense_ids = [ids[i] for i in np.argsort(-sims)[:pool]]
        sc = bm["model"].get_scores(bm25_tokens(r["query"]))
        lex_ids = [bm["ids"][i] for i in sorted(range(len(sc)), key=lambda i: -sc[i])[:pool]]
        ranked = [d for d, _ in sorted(_rrf([dense_ids, lex_ids]).items(),
                                       key=lambda kv: -kv[1])]
        if rerank and ranked:
            cand = ranked[:pool]
            scores = reranker(rerank_model).predict(
                [(r["query"], by_id[c]) for c in cand], batch_size=8, show_progress_bar=False)
            order = sorted(range(len(cand)), key=lambda i: -scores[i])
            ranked = [cand[i] for i in order] + ranked[pool:]
        gold = set(r["gold_ids"])
        rank = next((i + 1 for i, d in enumerate(ranked) if d in gold), None)
        nd = ndcg_at_k(ranked, gold, 10)
        ndcg += nd; rr += (1.0 / rank) if rank else 0.0
        for k in ks:
            if rank and rank <= k: hit[k] += 1
        p = per[r.get("kind", "?")]
        p["n"] += 1; p["ndcg"] += nd
        if rank and rank <= 10: p["hit10"] += 1
        if rank: p["ranks"].append(rank)
    N = len(rows)
    return dict(n=N, recall={k: hit[k] / N for k in ks}, mrr=rr / N, ndcg=ndcg / N,
                per={k: dict(n=v["n"], recall10=v["hit10"] / v["n"], ndcg=v["ndcg"] / v["n"],
                             median=sorted(v["ranks"])[len(v["ranks"]) // 2] if v["ranks"] else None)
                     for k, v in per.items()})

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/bge-m3-gita")
    ap.add_argument("--queries", default="finetune/held_queries.jsonl")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--rerank-model", default="BAAI/bge-reranker-v2-m3")
    a = ap.parse_args()
    r = run(a.model, a.queries, a.n, a.seed, rerank=a.rerank, rerank_model=a.rerank_model)
    print(f"\nmodel   : {a.model}")
    print(f"queries : {r['n']} from {a.queries}")
    print(f"  recall@1={r['recall'][1]:.3f}  recall@3={r['recall'][3]:.3f}  "
          f"recall@5={r['recall'][5]:.3f}  recall@10={r['recall'][10]:.3f}")
    print(f"  recall@20={r['recall'][20]:.3f} recall@50={r['recall'][50]:.3f}  <- ceiling for reranking")
    print(f"  NDCG@10={r['ndcg']:.3f}   MRR={r['mrr']:.3f}")
    for k, v in sorted(r["per"].items()):
        print(f"    {k:<9} n={v['n']:<4} recall@10={v['recall10']:.3f}  "
              f"NDCG@10={v['ndcg']:.3f}  median_rank={v['median']}")
    print(json.dumps(r, default=str))
