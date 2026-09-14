"""Hybrid retrieval over the Gita corpus: BGE-M3 dense + char-ngram BM25, fused with RRF.
Fully local — no API calls."""
from __future__ import annotations
import json, os, pickle, re, unicodedata
from pathlib import Path

ROOT       = Path(__file__).resolve().parent.parent
CORPUS     = ROOT / "corpus" / "gita.jsonl"
QDRANT_DIR = ROOT / "index" / "qdrant"
BM25_PATH  = ROOT / "index" / "bm25.pkl"
COLLECTION = "gita"
MODEL_NAME = "/Users/mobilemy/telugu-rag/models/bge-m3-gita"
RERANK_MODEL = ROOT / "models" / "bge-reranker-gita"
RERANK_MAX_LEN = 192          # measured: longest (query, doc) pair is 146 tokens
RERANK_POOL = 30     # Measured on 80 held-out queries, one rerank pass, pools
                     # derived by truncation (so pool is the only variable):
                     #   pool  recall@10  NDCG@10  ceiling  captured
                     #     15     0.562    0.436    0.588     96%
                     #     30     0.625    0.462    0.762     82%
                     #     50     0.625    0.467    0.812     77%
                     # recall@10 saturates at 30; pool=50 adds 40% compute for
                     # +0.005 NDCG. Candidates at RRF ranks 30-50 are ones the
                     # retriever already scored poorly and the reranker rarely
                     # rescues - the ceiling rises faster than it can climb.
DIM        = 1024

TELUGU = re.compile(r"[ఀ-౿]")
LATIN  = re.compile(r"[A-Za-z]")

def norm(s: str) -> str:
    return unicodedata.normalize("NFC", s or "").strip()

def bm25_tokens(s: str) -> list[str]:
    """Char 4-grams + whitespace tokens. Telugu is agglutinative, so word-level
    BM25 misses; char n-grams survive sandhi and inflection."""
    s = norm(s)
    grams = [s[i:i+4] for i in range(max(0, len(s) - 3))]
    return grams + [w for w in s.split() if w]

def load_corpus() -> list[dict]:
    return [json.loads(l) for l in CORPUS.open(encoding="utf-8")]

def enrichment_text(r: dict) -> str:
    """The thematic surface: themes, life situations, emotions - all Telugu.

    Indexed as its OWN vector rather than appended to the tatparyam. The
    bi-encoder was fine-tuned on (query -> tatparyam) pairs; appending changes
    the text it learned to represent and risks the lookup/tenglish gains.
    """
    parts = (r.get("themes_te") or []) + (r.get("life_situations_te") or []) \
            + (r.get("emotions_te") or [])
    return norm(" . ".join(p for p in parts if p))

def retrieval_text(r: dict) -> str:
    """What we embed. The tatparyam is modern Telugu prose — same register as
    user queries — so it is the primary retrieval surface. Enrichment fields are
    appended when present."""
    parts = [r.get("tatparyam_te", "")]
    if r.get("themes"):         parts.append(" ".join(r["themes"]))
    if r.get("life_situations"):parts.append(" ".join(r["life_situations"]))
    parts.append(r.get("adhyaya", ""))
    return norm(" ".join(p for p in parts if p))

# ------------------------------------------------------------------ embedding
_model = None
def encoder():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME, device="cpu")
        _model.max_seq_length = 512
    return _model

def embed(texts: list[str], batch_size: int = 4) -> list[list[float]]:
    return encoder().encode(
        texts, batch_size=batch_size, normalize_embeddings=True,
        show_progress_bar=True, convert_to_numpy=True,
    ).tolist()

# --------------------------------------------------------------------- search
_qc = None
def _client():
    """Singleton: Qdrant local mode takes an exclusive lock on the directory,
    so a per-call client deadlocks on the second query."""
    global _qc
    if _qc is None:
        from qdrant_client import QdrantClient
        _qc = QdrantClient(path=str(QDRANT_DIR))
    return _qc

def _rrf(ranked_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for lst in ranked_lists:
        for rank, doc_id in enumerate(lst):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return scores

RERANK_BOTH_SURFACES = True   # score (query, tatparyam) and (query, enrichment),
                              # take the max. Doubles cross-encoder cost.

_enr = None
def enrichment_map() -> dict:
    """verse_id -> thematic text, for reranking the surface a verse was found by."""
    global _enr
    if _enr is None:
        f = ROOT / "corpus" / "enrichment.jsonl"
        _enr = {}
        if f.exists():
            import json as _j
            for line in f.open(encoding="utf-8"):
                if line.strip():
                    r = _j.loads(line)
                    t = enrichment_text(r)
                    if t: _enr[r["verse_id"]] = t
    return _enr

_ce = None
def reranker():
    """Cross-encoder, lazily loaded. Returns None if the model is not present,
    so the pipeline degrades to retrieval-only rather than failing."""
    global _ce
    if _ce is None:
        if not (RERANK_MODEL / "config.json").exists():
            return None
        from sentence_transformers import CrossEncoder
        _ce = CrossEncoder(str(RERANK_MODEL), max_length=RERANK_MAX_LEN, device="cpu")
    return _ce

_bm = None
def _bm25():
    global _bm
    if _bm is None:
        with BM25_PATH.open("rb") as f:
            _bm = pickle.load(f)
    return _bm

def search(query: str, top_k: int = 8, pool: int = RERANK_POOL,
           adhyaya: int | None = None, rerank: bool = True) -> list[dict]:
    """Dense + BM25 fused with RRF, then optionally reordered by a cross-encoder.

    On held-out queries the cross-encoder lifts recall@10 0.512 -> 0.634 and
    NDCG@10 0.358 -> 0.468, with the largest gain on romanized (tenglish) input.
    """
    from qdrant_client import models as qm
    query = norm(query)
    client = _client()

    qflt = None
    if adhyaya is not None:
        qflt = qm.Filter(must=[qm.FieldCondition(
            key="adhyaya_no", match=qm.MatchValue(value=adhyaya))])

    qvec = embed([query], batch_size=1)[0]
    hits = client.query_points(COLLECTION, query=qvec, limit=pool * 2,
                               query_filter=qflt, with_payload=True).points
    dense_ids, payloads = [], {}
    for h in hits:                      # keep first (best) hit per verse
        vid = h.payload["id"]
        if vid not in payloads:
            payloads[vid] = h.payload; dense_ids.append(vid)
    dense_ids = dense_ids[:pool]

    bm = _bm25()
    scores = bm["model"].get_scores(bm25_tokens(query))
    order  = sorted(range(len(scores)), key=lambda i: -scores[i])[:pool]
    lex_ids = [bm["ids"][i] for i in order]
    for i in order:
        payloads.setdefault(bm["ids"][i], bm["payloads"][i])

    fused = _rrf([dense_ids, lex_ids])
    if adhyaya is not None:
        fused = {k: v for k, v in fused.items()
                 if payloads.get(k, {}).get("adhyaya_no") == adhyaya}
    ranked = sorted(fused.items(), key=lambda kv: -kv[1])
    ce = reranker() if rerank else None
    if ce is not None and ranked:
        cand = [d for d, _ in ranked[:pool]]
        scores = list(ce.predict([(query, payloads[c].get("tatparyam_te", "")) for c in cand],
                                 batch_size=32, show_progress_bar=False))
        if RERANK_BOTH_SURFACES:
            # A verse found via its thematic vector still had to win on its
            # tatparyam - the very surface that failed to retrieve it. Score the
            # enrichment text too and keep the better of the two.
            em = enrichment_map()
            idx = [i for i, c in enumerate(cand) if c in em]
            if idx:
                es = ce.predict([(query, em[cand[i]]) for i in idx],
                                batch_size=32, show_progress_bar=False)
                for k, i in enumerate(idx):
                    scores[i] = max(scores[i], float(es[k]))
        order = sorted(range(len(cand)), key=lambda i: -scores[i])
        out = []
        for i in order[:top_k]:
            p = dict(payloads[cand[i]])
            p["_score"] = round(float(scores[i]), 5); p["_reranked"] = True
            out.append(p)
        return out
    out = []
    for doc_id, score in ranked[:top_k]:
        p = dict(payloads[doc_id]); p["_score"] = round(score, 5); p["_reranked"] = False
        out.append(p)
    return out
