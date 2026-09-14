"""Generate a thematic layer per verse, in Telugu.

Thematic queries fail because the query lives at the level of a life situation
("నా ఉద్యోగంలో గుర్తింపు రావడం లేదు") while the verse lives at the level of a
teaching. Nothing bridges them. This builds the bridge at index time.

The output MUST be Telugu: it is matched against Telugu queries. Generating
English themes would recreate the same gap one level up.
"""
from __future__ import annotations
import json, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
MODEL = "claude-haiku-4-5"
BATCH, WORKERS = 8, 4

SYSTEM = """You build a thematic index for the Bhagavad Gita so that people can find
verses by describing their own life, not by knowing the text.

For each verse you get its Telugu tatparyam. Produce, IN TELUGU:

- themes_te: 3-5 short abstract themes this verse speaks to
  (e.g. "ఫలాపేక్ష", "కర్తవ్యం", "కోపం", "అహంకారం", "ఓర్పు", "గుర్తింపు").
- life_situations_te: 3-4 concrete everyday situations, written the way an
  ordinary person would describe their own problem - first person, plain
  వ్యావహారిక Telugu, no scripture vocabulary. These are matched against what
  users actually type, so phrase them as a person would.
  Good: "నా పని ఎవరూ గుర్తించడం లేదు"
  Bad:  "నిష్కామ కర్మ యోగ సాధన"
- emotions_te: 2-3 feelings someone in that situation would have
  (e.g. "నిరాశ", "కోపం", "భయం", "అసూయ", "దుఃఖం").

Only what the verse genuinely supports. Do not stretch it to fit a situation
it does not speak to - a wrong mapping is worse than no mapping."""

class VerseEnrichment(BaseModel):
    verse_id: str
    themes_te: list[str] = Field(min_length=1, max_length=8)
    life_situations_te: list[str] = Field(min_length=1, max_length=6)
    emotions_te: list[str] = Field(min_length=0, max_length=5)

class Batch(BaseModel):
    items: list[VerseEnrichment]

import re
# Observed once (verse 1.2): Bengali characters spliced into Telugu words
# ("ప্রతిస্పందన"). Contaminated strings would be embedded as-is, so drop them.
NON_TELUGU = re.compile(r"[\u0900-\u097F\u0980-\u09FF\u0B80-\u0BFF\u0C80-\u0CFF]")
def clean_list(xs):
    return [x for x in xs if x and not NON_TELUGU.search(x)]

_lock = threading.Lock(); USAGE = {"in": 0, "out": 0}
_client = None
def client():
    global _client
    if _client is None:
        from anthropic import Anthropic
        _client = Anthropic()
    return _client

def gen(chunk):
    try:
        return _gen(chunk)
    except Exception as e:
        print(f"  batch failed ({type(e).__name__}); retrying singly", flush=True)
        out = []
        for v in chunk:
            try: out += _gen([v])
            except Exception as e2: print(f"    skip {v['id']}: {type(e2).__name__}", flush=True)
        return out

def _gen(chunk):
    payload = "\n\n".join(f"verse_id: {r['id']}\ntatparyam: {r['tatparyam_te']}" for r in chunk)
    r = client().messages.parse(
        model=MODEL, max_tokens=8000, system=SYSTEM,
        messages=[{"role": "user", "content": payload}], output_format=Batch)
    with _lock:
        USAGE["in"] += r.usage.input_tokens; USAGE["out"] += r.usage.output_tokens
    out = []
    for it in r.parsed_output.items:
        d = it.model_dump()
        for k in ("themes_te", "life_situations_te", "emotions_te"):
            before = len(d[k]); d[k] = clean_list(d[k])
            if len(d[k]) < before:
                d.setdefault("dropped", 0)
                d["dropped"] += before - len(d[k])
        if d["themes_te"] and d["life_situations_te"]:
            out.append(d)
    return out

def main():
    src = ROOT / "corpus" / "gita.jsonl"
    dst = ROOT / "corpus" / "enrichment.jsonl"
    verses = [json.loads(l) for l in src.open(encoding="utf-8") if l.strip()]
    done = set()
    if dst.exists():
        done = {json.loads(l)["verse_id"] for l in dst.open(encoding="utf-8") if l.strip()}
    todo = [v for v in verses if v["id"] not in done]
    if not todo:
        print(f"nothing to do ({len(done)} already enriched)"); return
    chunks = [todo[i:i+BATCH] for i in range(0, len(todo), BATCH)]
    print(f"{len(todo)} verses in {len(chunks)} calls ({len(done)} cached)")
    n = 0
    with dst.open("a", encoding="utf-8") as f, ThreadPoolExecutor(WORKERS) as ex:
        for rows in ex.map(gen, chunks):
            with _lock:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush(); n += len(rows)
                if n % 80 == 0: print(f"  {n}/{len(todo)}", flush=True)
    cost = USAGE["in"]*1/1e6 + USAGE["out"]*5/1e6
    print(f"enriched {n} verses  |  tokens in={USAGE['in']:,} out={USAGE['out']:,}  ~${cost:.2f}")

if __name__ == "__main__":
    main()
