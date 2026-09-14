"""Generate synthetic Telugu training queries for each verse, with Claude.

One record per (query, verse) pair. Resumable: already-generated verse ids are
skipped, so an interrupted run costs nothing.
"""
from __future__ import annotations
import json, os, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
MODEL = "claude-opus-5"
BATCH = 8          # verses per API call
WORKERS = 4

SYSTEM = """You write realistic Telugu search queries for a Bhagavad Gita retrieval system.

For each verse you are given its Telugu tatparyam (prose meaning). Write questions
that a real person would type and that THIS verse specifically answers.

Rules:
- Modern spoken Telugu (వ్యావహారికం). Never classical/grandhika forms.
- Do NOT quote the verse's wording. Use the words an ordinary person would choose,
  not the commentary's vocabulary. Paraphrase the concept.
- `lookup`: 2 direct factual questions ("గీత ఏమి చెబుతుంది...", "ఎలా...", "ఎవరు...").
- `thematic`: 2 first-person life-situation questions. A person describing a real
  problem in their life that this verse speaks to. Do not mention the Gita.
  Example shape: "నాకు పని దగ్గర గుర్తింపు రావడం లేదు, ఏం చేయాలి?"
- `tenglish`: 1 question in romanized Telugu, as people actually type on phones
  (no diacritics, casual spelling). Example: "karma phalam gurinchi emi cheppadu"
- Every question must be answerable from that verse alone."""

class VerseQueries(BaseModel):
    verse_id: str
    lookup: list[str] = Field(min_length=2, max_length=2)
    thematic: list[str] = Field(min_length=2, max_length=2)
    tenglish: str

class Batch(BaseModel):
    items: list[VerseQueries]

_lock = threading.Lock()
USAGE = {'in':0,'out':0}
_client = None
def client():
    global _client
    if _client is None:
        from anthropic import Anthropic
        _client = Anthropic()
    return _client

def gen(chunk: list[dict]) -> list[dict]:
    payload = "\n\n".join(
        f"verse_id: {r['id']}\ncitation: {r['citation_te']}\ntatparyam: {r['tatparyam_te']}"
        for r in chunk)
    r = client().messages.parse(
        model=MODEL, max_tokens=8000, system=SYSTEM,
        messages=[{"role": "user", "content": payload}],
        output_format=Batch,
        output_config={"effort": "medium"},
        thinking={"type": "adaptive"},
    )
    with _lock:
        USAGE['in'] += r.usage.input_tokens; USAGE['out'] += r.usage.output_tokens
    by_id = {x["id"]: x for x in chunk}
    out = []
    for it in r.parsed_output.items:
        v = by_id.get(it.verse_id)
        if not v: continue
        for q in it.lookup:    out.append(dict(query=q, verse_id=v["id"], kind="lookup"))
        for q in it.thematic:  out.append(dict(query=q, verse_id=v["id"], kind="thematic"))
        out.append(dict(query=it.tenglish, verse_id=v["id"], kind="tenglish"))
    return out

def run(src: Path, dst: Path, limit: int | None = None):
    verses = [json.loads(l) for l in src.open(encoding="utf-8")]
    done = set()
    if dst.exists():
        done = {json.loads(l)["verse_id"] for l in dst.open(encoding="utf-8") if l.strip()}
    todo = [v for v in verses if v["id"] not in done]
    if limit: todo = todo[:limit]
    if not todo:
        print(f"{dst.name}: nothing to do ({len(done)} verses already generated)"); return
    chunks = [todo[i:i+BATCH] for i in range(0, len(todo), BATCH)]
    print(f"{dst.name}: {len(todo)} verses in {len(chunks)} calls ({len(done)} cached)")
    n = 0
    with dst.open("a", encoding="utf-8") as f, ThreadPoolExecutor(WORKERS) as ex:
        for res in ex.map(lambda c: (c, gen(c)), chunks):
            chunk, rows = res
            with _lock:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                n += len(rows)
                print(f"  +{len(rows):>2} pairs  (total {n})", flush=True)
    cost = USAGE['in']*5/1e6 + USAGE['out']*25/1e6
    print(f"{dst.name}: wrote {n} pairs")
    print(f"  tokens in={USAGE['in']:,} out={USAGE['out']:,}  approx cost ${cost:.2f}")

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "train"
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else None
    src = ROOT / "finetune" / f"{which}_verses.jsonl"
    dst = ROOT / "finetune" / f"{which}_queries.jsonl"
    run(src, dst, lim)
