"""Pure functions with no model or network dependency.

Extracted so the invariants that actually broke in development are testable in
milliseconds: citation id parsing, verbatim anchoring, Telugu tokenisation and
rank fusion.
"""
from __future__ import annotations
import math, re, unicodedata

TELUGU  = re.compile(r"[ఀ-౿]")
LATIN   = re.compile(r"[A-Za-z]")
# Devanagari / Bengali / Tamil / Kannada / Latin-1 — scripts a Telugu answer
# should never contain. Smaller models emit these mid-word (మనुష్యుడు, ప్రశాంáfrica).
FOREIGN = re.compile(r"[À-ɏЀ-ӿऀ-ॿঀ-৿"
                     r"஀-௿ಀ-೿]")
VERSE_ID = re.compile(r"(?:gita[-\s]*)?(\d{1,2})[-.](\d{1,3})")
SENT_SPLIT = re.compile(r"(?<!\d)[.।!?\n]+(?!\d)")

def norm(s: str) -> str:
    return unicodedata.normalize("NFC", s or "").strip()

def squash(s: str) -> str:
    return re.sub(r"\s+", "", norm(s))

def parse_verse_id(raw: str) -> str | None:
    """Accept every form the model emits: 'gita-2-47', '[gita-2-47]', '2-47', '2.47'."""
    m = VERSE_ID.search(raw or "")
    return f"gita-{m.group(1)}-{m.group(2)}" if m else None

def bm25_tokens(s: str) -> list[str]:
    """Char 4-grams plus whitespace tokens. Telugu is agglutinative and sandhi
    fuses word boundaries, so word-level BM25 misses the stem."""
    s = norm(s)
    return [s[i:i+4] for i in range(max(0, len(s) - 3))] + [w for w in s.split() if w]

def rrf(ranked_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal rank fusion. Fuses by RANK because cosine and BM25 scores are
    not on a comparable scale."""
    out: dict[str, float] = {}
    for lst in ranked_lists:
        for i, d in enumerate(lst):
            out[d] = out.get(d, 0.0) + 1.0 / (k + i + 1)
    return out

def longest_verbatim_run(quote: str, source: str) -> int:
    q, src = squash(quote), squash(source)
    best = 0
    for i in range(len(q)):
        for j in range(len(q), i + best, -1):
            if q[i:j] in src:
                best = max(best, j - i)
                break
    return best

def anchor_span(quote: str, source: str, min_run: int = 24, width: int = 240) -> str | None:
    """Confirm a quote points at real source text, then return the SOURCE span.

    Exact matching rejected citations over a single dropped character
    ('ఫలాలపై' vs 'ఫలాలపైన'). Requiring a long verbatim run instead tolerates
    transcription drift while still rejecting fabrication — and because the
    SOURCE text is returned, nothing shown to a user originates from the model.
    """
    if longest_verbatim_run(quote, source) < min_run:
        return None
    q, src = squash(quote), squash(source)
    best, pos = 0, 0
    for i in range(len(q)):
        for j in range(len(q), i + best, -1):
            if q[i:j] in src:
                if j - i > best:
                    best, pos = j - i, src.index(q[i:j])
                break
    seen = start = 0
    for k, ch in enumerate(source):
        if not ch.isspace():
            if seen == pos:
                start = k
                break
            seen += 1
    tail = source[start:]
    return (tail[:width].rstrip() + ("…" if len(tail) > width else "")) or None

def ndcg_at_k(ranked: list[str], gold: set[str], k: int = 10) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in gold)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal if ideal else 0.0
