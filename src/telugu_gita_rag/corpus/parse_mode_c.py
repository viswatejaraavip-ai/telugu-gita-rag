"""Mode C: chapters where each verse block is
     [incipit]
     N. <sanskrit line 1> |
     <sanskrit line 2> ||
     <telugu tatparyam...>
Explicit verse numbers make these citations trustworthy, unlike mode B.
"""
import re, urllib.parse
from build_corpus import wikitext, strip_markup

NUMLINE = re.compile(r"^(\d{1,3})\.\s*(.*)$")
TE      = re.compile(r"[ఀ-౿]")
TERM    = "।.?!:"

def is_sanskrit_line(l): return l.rstrip().endswith(("|", "||"))

def detect(body):
    """Mode C if several lines start with 'N.' and carry a danda."""
    n = sum(1 for l in body.split("\n")
            if NUMLINE.match(l.strip()) and "|" in l)
    return n >= 5

def parse_C(body, ch, name, rec):
    lines = [l.strip() for l in body.split("\n") if l.strip()]
    starts = [i for i, l in enumerate(lines) if NUMLINE.match(l)]
    out = []
    for bi, si in enumerate(starts):
        end = starts[bi + 1] if bi + 1 < len(starts) else len(lines)
        block = lines[si:end]
        # the trailing line of a block is the NEXT verse's incipit - drop it
        while len(block) > 1:
            last = block[-1]
            if (len(last) < 46 and not last.rstrip().endswith(tuple(TERM))
                    and not is_sanskrit_line(last)):
                block = block[:-1]
            else:
                break
        num = int(NUMLINE.match(block[0]).group(1))
        head = NUMLINE.match(block[0]).group(2)
        sa, te = [], []
        for j, l in enumerate(block):
            txt = head if j == 0 else l
            if not txt: continue
            if is_sanskrit_line(txt) or (j == 0 and len(txt) < 40):
                sa.append(txt)
            else:
                te.append(txt)
        tat = re.sub(r"\s+", " ", " ".join(te)).strip()
        if len(TE.findall(tat)) < 15: continue
        out.append(rec(ch, name, num,
                       re.sub(r"\s+", " ", " ".join(sa)).strip(" |।॥"),
                       tat, "C"))
    return out

if __name__ == "__main__":
    from gita import CHAPTERS, EXPECTED, rec
    print(f"{'ch':>3} {'name':<30}{'mode':>5}{'got':>5}{'exp':>5}")
    modec = []
    for n, name in CHAPTERS:
        body = strip_markup(wikitext(name))
        body = re.sub(r"__[A-Z]+__", " ", body)
        if detect(body):
            v = parse_C(body, n, name, rec)
            modec.append(n)
            ok = "OK" if len(v) == EXPECTED[n] else "** CHECK"
            print(f"{n:>3} {name:<30}{'C':>5}{len(v):>5}{EXPECTED[n]:>5}  {ok}")
    print(f"\nmode-C chapters: {modec}")
