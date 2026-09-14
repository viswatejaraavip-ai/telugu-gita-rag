"""Build the Qdrant collection + BM25 index from corpus/gita.jsonl."""
import pickle, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import json
from retrieval import (BM25_PATH, COLLECTION, DIM, QDRANT_DIR, bm25_tokens,
                       embed, load_corpus, retrieval_text, enrichment_text, ROOT)

def main(rebuild: bool = True):
    recs = load_corpus()
    print(f"corpus: {len(recs)} verses")
    texts = [retrieval_text(r) for r in recs]

    print("embedding with BGE-M3 on CPU (this is the slow part)...")
    vecs = embed(texts)
    print(f"  -> {len(vecs)} vectors x {len(vecs[0])} dims")

    from qdrant_client import QdrantClient, models as qm
    if rebuild and QDRANT_DIR.exists():
        shutil.rmtree(QDRANT_DIR)
    QDRANT_DIR.parent.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(QDRANT_DIR))
    client.recreate_collection(
        COLLECTION,
        vectors_config=qm.VectorParams(size=DIM, distance=qm.Distance.COSINE),
    )
    points = [qm.PointStruct(id=i, vector=vecs[i], payload=recs[i])
              for i in range(len(recs))]

    # Enrichment vectors: same payload (so the verse id resolves identically),
    # different text. A thematic query can match the life-situation phrasing
    # while a lookup query still matches the tatparyam.
    enr_path = ROOT / "corpus" / "enrichment.jsonl"
    if enr_path.exists():
        enr = {j["verse_id"]: j for j in
               (json.loads(l) for l in enr_path.open(encoding="utf-8") if l.strip())}
        by_id = {r["id"]: r for r in recs}
        pairs = [(vid, enrichment_text(e)) for vid, e in enr.items()
                 if vid in by_id and enrichment_text(e)]
        print(f"embedding {len(pairs)} enrichment vectors...")
        evecs = embed([t for _, t in pairs])
        for k, (vid, _) in enumerate(pairs):
            pl = dict(by_id[vid]); pl["_via"] = "enrichment"
            points.append(qm.PointStruct(id=len(recs) + k, vector=evecs[k], payload=pl))
    client.upsert(COLLECTION, points=points)
    print(f"  qdrant: {client.count(COLLECTION).count} points -> {QDRANT_DIR}")
    client.close()

    from rank_bm25 import BM25Okapi
    corpus_tokens = [bm25_tokens(t) for t in texts]
    bm = BM25Okapi(corpus_tokens)
    BM25_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BM25_PATH.open("wb") as f:
        pickle.dump({"model": bm, "ids": [r["id"] for r in recs],
                     "payloads": recs}, f)
    print(f"  bm25:   {len(corpus_tokens)} docs -> {BM25_PATH}")

if __name__ == "__main__":
    main()
