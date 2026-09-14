"""Post-parse cleanup of corpus/gita.jsonl.

Two defects the eval surfaced:
  1. Chapter 1 carries Shankara/Ramanuja/Madhva bhashya text inside tatparyam_te.
  2. Some verses have the NEXT verse's Sanskrit incipit/sloka appended.

Both are truncation problems: the real Telugu gloss is a prefix, and the junk is
a suffix that starts at a recognisable marker. We cut rather than re-parse, so
the 14 chapters whose verse counts match canon are not disturbed.
"""
import json, re, sys
from pathlib import Path

CORPUS = Path(__file__).resolve().parent.parent / "corpus" / "gita.jsonl"
BHASHYA = re.compile(r"(ఆదిశంకరాచార్యుల|రామానుజాచార్యుల|మధ్వాచార్యుల|భాష్యా?ల?ు?\s*:)")
TERM = "।.?!"

def clean(t: str) -> str:
    orig = t
    m = BHASHYA.search(t)
    if m:
        t = t[:m.start()]
    if "|" in t:
        head = t[:t.index("|")]
        cut = max(head.rfind(c) for c in TERM)
        t = head[:cut + 1] if cut > 0 else head
    t = re.sub(r"\s+", " ", t).strip()
    return t if len(t) >= 20 else orig.strip()

def main():
    recs = [json.loads(l) for l in CORPUS.open(encoding="utf-8")]
    changed, shrunk = 0, 0
    for r in recs:
        before = r["tatparyam_te"]
        after = clean(before)
        if after != before:
            changed += 1
            shrunk += len(before) - len(after)
            r["tatparyam_te"] = after
            r["cleaned"] = True
    # Deduplicate (adhyaya, sloka): the source repeats some verses, once with a
    # real Telugu gloss and once with only bhashya boilerplate. Keep the richer.
    best = {}
    for r in recs:
        k = (r["adhyaya_no"], r["sloka_no"])
        prev = best.get(k)
        score = len(r["tatparyam_te"]) - (10000 if BHASHYA.search(r["tatparyam_te"]) else 0)
        if prev is None or score > prev[0]:
            best[k] = (score, r)
    deduped = [r for _, r in sorted(best.values(), key=lambda x: (x[1]["adhyaya_no"], x[1]["sloka_no"]))]
    if len(deduped) != len(recs):
        print(f"deduplicated {len(recs) - len(deduped)} repeated verse(s)")
    recs = deduped

    with CORPUS.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"cleaned {changed} records, removed {shrunk:,} chars of junk")
    print(f"mean tatparyam now: {sum(len(r['tatparyam_te']) for r in recs)//len(recs)} chars")
    left_b = sum(1 for r in recs if BHASHYA.search(r["tatparyam_te"]))
    left_d = sum(1 for r in recs if "|" in r["tatparyam_te"])
    print(f"remaining bhashya contamination: {left_b}")
    print(f"remaining danda bleed:           {left_d}")

if __name__ == "__main__":
    main()
