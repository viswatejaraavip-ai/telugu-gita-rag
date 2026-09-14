#!/usr/bin/env python3
"""Build a verse-aligned Telugu corpus (Valmiki Ramayana + Bhagavad Gita)
from te.wikisource.org.  Output: JSONL, one record per verse."""
import json, os, re, sys, time, urllib.parse, urllib.request

API   = "https://te.wikisource.org/w/api.php"
UA    = "TeluguRAG-corpus/0.1 (research; contact: viswa@tejaswiservices.com)"
CACHE = "cache"; OUT = "corpus"
os.makedirs(CACHE, exist_ok=True); os.makedirs(OUT, exist_ok=True)

TE_DIGITS = str.maketrans("౦౧౨౩౪౫౬౭౮౯", "0123456789")
TE_RANGE  = re.compile(r"[ఀ-౿]")

def api(**p):
    p.setdefault("format", "json")
    url = API + "?" + urllib.parse.urlencode(p)
    for a in range(5):
        try:
            r = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(r, timeout=40) as f:
                return json.loads(f.read().decode())
        except Exception as e:
            if a == 4:
                print("  FAIL", e, file=sys.stderr); return {}
            time.sleep(2 * (a + 1))
    return {}

def wikitext(title):
    """Fetch page wikitext, cached on disk."""
    key = os.path.join(CACHE, re.sub(r"[^\w]", "_", title)[:120] + ".txt")
    if os.path.exists(key):
        return open(key, encoding="utf-8").read()
    d = api(action="parse", page=title, prop="wikitext", redirects=1)
    w = d.get("parse", {}).get("wikitext", {}).get("*", "")
    open(key, "w", encoding="utf-8").write(w)
    time.sleep(0.25)
    return w

def allpages(prefix):
    out, cont = [], {}
    while True:
        d = api(action="query", list="allpages", apprefix=prefix, aplimit=500,
                apnamespace=0, apfilterredir="nonredirects", **cont)
        out += [p["title"] for p in d.get("query", {}).get("allpages", [])]
        if "continue" in d: cont = d["continue"]; time.sleep(0.3)
        else: return out

def strip_markup(t):
    t = re.sub(r"\{\{[^{}]*\}\}", " ", t)
    t = re.sub(r"\{\{[^{}]*\}\}", " ", t)          # nested pass
    t = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", t)
    t = re.sub(r"</?(poem|div|span|center|br|p|b|i)[^>]*>", "\n", t, flags=re.I)
    t = re.sub(r"<ref[^>]*>.*?</ref>", " ", t, flags=re.S|re.I)
    t = re.sub(r"<!--.*?-->", " ", t, flags=re.S)
    t = re.sub(r"^[=']+\s*|\s*[=']+$", "", t, flags=re.M)
    t = t.replace("&nbsp;", " ")
    return t

# ---------------------------------------------------------------- Ramayana
REF = re.compile(r"[|\[]\s*([\d౦-౯]{1,2})\s*[-–]\s*([\d౦-౯]{1,3})\s*[-–]\s*([\d౦-౯]{1,3})[a-z]?\s*[|\]]")

def parse_ramayana(title, kanda, kanda_no, sarga_no):
    """Sanskrit sloka ends with |k-s-v| ; the Telugu gloss follows, tagged [k-s-v]."""
    body = strip_markup(wikitext(title))
    if not body.strip(): return []
    parts, last = [], 0
    for m in REF.finditer(body):
        seg = body[last:m.start()].strip()
        ref = tuple(int(g.translate(TE_DIGITS)) for g in m.groups())
        closer = m.group(0)[-1]
        parts.append((ref, closer, seg))
        last = m.end()
    recs, pending = [], {}
    for ref, closer, seg in parts:
        seg = re.sub(r"\s*\n\s*", " ", seg).strip(" |।॥")
        if not seg: continue
        if closer == "|":                       # end of a Sanskrit sloka
            pending[ref] = seg
        else:                                   # end of a Telugu gloss
            sl = pending.pop(ref, "")
            recs.append(dict(sloka=sl, tatparyam=seg, ref=ref))
    out = []
    for r in recs:
        k, s, v = r["ref"]
        tat = r["tatparyam"]
        if len(TE_RANGE.findall(tat)) < 10: continue
        out.append({
            "id": f"ram-{kanda_no}-{s}-{v}",
            "text_type": "verse", "epic": "Valmiki Ramayana",
            "kanda": kanda, "kanda_no": kanda_no, "sarga": sarga_no, "sloka": v,
            "citation_te": f"వాల్మీకి రామాయణము, {kanda}, సర్గ {sarga_no}, శ్లోకం {v}",
            "citation_en": f"Valmiki Ramayana {kanda_no}.{sarga_no}.{v}",
            "sloka_sa": r["sloka"], "tatparyam_te": tat,
            "source_page": title,
            "source_url": "https://te.wikisource.org/wiki/" + urllib.parse.quote(title.replace(" ", "_")),
            "license": "CC BY-SA 4.0",
        })
    return out

# -------------------------------------------------------------------- Gita
GREF = re.compile(r"\|\|\s*(\d{1,2})\s*[-–]\s*(\d{1,3})\s*\|\|")

def parse_gita(title, ch_no, ch_name):
    body = strip_markup(wikitext(title))
    if not body.strip(): return []
    chunks, last, prev = [], 0, None
    for m in GREF.finditer(body):
        chunks.append((prev, body[last:m.start()], m.group(0)))
        prev = (int(m.group(1)), int(m.group(2))); last = m.end()
    chunks.append((prev, body[last:], None))
    out = []
    for i, (ref, seg, _) in enumerate(chunks):
        if ref is None: continue
        ch, vs = ref
        sloka = re.sub(r"\s*\n\s*", " ", chunks[i-1][1] if i else "").strip()
        tail  = seg.strip()
        bh    = ""
        if "భాష్య" in tail:
            tail, bh = re.split(r"భాష్యాల?ు?\s*:?", tail, maxsplit=1)[0], \
                       re.split(r"భాష్యాల?ు?\s*:?", tail, maxsplit=1)[1]
        tat = re.sub(r"\s*\n\s*", " ", tail).strip()
        if len(TE_RANGE.findall(tat)) < 8: continue
        out.append({
            "id": f"gita-{ch}-{vs}", "text_type": "verse", "epic": "Bhagavad Gita",
            "adhyaya": ch_name, "adhyaya_no": ch, "sloka": vs,
            "citation_te": f"భగవద్గీత, {ch_name}, శ్లోకం {ch}.{vs}",
            "citation_en": f"Bhagavad Gita {ch}.{vs}",
            "sloka_sa": sloka, "tatparyam_te": tat,
            "bhashya_te": re.sub(r"\s+", " ", bh).strip()[:4000],
            "source_page": title,
            "source_url": "https://te.wikisource.org/wiki/" + urllib.parse.quote(title.replace(" ", "_")),
            "license": "CC BY-SA 4.0",
        })
    return out

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "sample"
    if mode == "sample":
        r = parse_ramayana("బాలకాండము - సర్గము 40", "బాలకాండము", 1, 40)
        print(f"Ramayana sample -> {len(r)} verses"); print(json.dumps(r[0], ensure_ascii=False, indent=2)[:900])
        g = parse_gita("కర్మ యోగము", 3, "కర్మ యోగము")
        print(f"\nGita sample -> {len(g)} verses"); print(json.dumps(g[0], ensure_ascii=False, indent=2)[:900])
