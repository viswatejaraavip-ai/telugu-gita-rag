"""Telugu Bhagavad Gita RAG — Hugging Face Space (GPU).

Answers only from the 699-verse corpus, and every quote is verified verbatim
against its source verse before it reaches the user.

No Qdrant here: 699 verses x 2 surfaces is ~1400 vectors, so a numpy matmul is
faster than any index and sidesteps Qdrant local-mode's single-process file lock
(which would break a multi-worker server).
"""
from __future__ import annotations
import json, math, os, re, unicodedata
from pathlib import Path
from typing import Literal, Optional

import gradio as gr
import numpy as np
import spaces          # ZeroGPU
import torch
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

ROOT = Path(__file__).parent

BIENCODER = os.environ.get("BIENCODER_REPO", "BAAI/bge-m3")
RERANKER  = os.environ.get("RERANKER_REPO",  "BAAI/bge-reranker-v2-m3")
MODEL_FAST = os.environ.get("MODEL_FAST", "claude-haiku-4-5")   # route / transliterate
MODEL_GEN  = os.environ.get("MODEL_GEN",  "claude-haiku-4-5")   # answer generation
RERANK_POOL = 30
RERANK_MAX_LEN = 192

# Which enrichment file to index. "selective" keeps only verses judged to have a
# life application (352 of 699); "full" keeps all. Measured caveat: on held-out
# thematic queries, 55% of gold verses fall outside the selective keep-set, so
# selective may cost thematic recall. Flip to "full" to compare.
ENRICHMENT_VARIANT = os.environ.get("ENRICHMENT_VARIANT", "filtered")  # full | filtered | off

TELUGU = re.compile(r"[ఀ-౿]"); LATIN = re.compile(r"[A-Za-z]")
SENT_SPLIT = re.compile(r"(?<!\d)[.।!?\n]+(?!\d)")
def norm(s): return unicodedata.normalize("NFC", s or "").strip()
def squash(s): return re.sub(r"\s+", "", norm(s))

def bm25_tokens(s):
    s = norm(s)
    return [s[i:i+4] for i in range(max(0, len(s)-3))] + [w for w in s.split() if w]

def jl(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

print("loading corpus...")
CORPUS = jl(ROOT/"data"/"gita.jsonl")
BY_ID  = {r["id"]: r for r in CORPUS}
IDS    = [r["id"] for r in CORPUS]
ENR = {r["verse_id"]: r for r in jl(ROOT/"data"/"enrichment.jsonl")}

def retrieval_text(r): return norm(" ".join(x for x in [r.get("tatparyam_te",""), r.get("adhyaya","")] if x))
def enrichment_text(r):
    parts=(r.get("themes_te") or [])+(r.get("life_situations_te") or [])+(r.get("emotions_te") or [])
    return norm(" . ".join(p for p in parts if p))

ENR_IDS   = [i for i in IDS if i in ENR and enrichment_text(ENR[i])]
print(f"corpus={len(CORPUS)}  enrichment file loaded={len(ENR)}")

print("loading models...")
BI = SentenceTransformer(BIENCODER); BI.max_seq_length = 512
CE = CrossEncoder(RERANKER, max_length=RERANK_MAX_LEN)

# Corpus vectors are PRECOMPUTED and shipped. ZeroGPU boots on CPU, so embedding
# ~1400 texts at every restart would cost minutes per cold start. Only the query
# is embedded at request time, inside the @spaces.GPU block.
V_TAT = np.load(ROOT/"data"/"vec_tat.npy")
_E_ALL = np.load(ROOT/"data"/"vec_enr.npy")
_E_IDS = json.load(open(ROOT/"data"/"vec_enr_ids.json", encoding="utf-8"))
if ENRICHMENT_VARIANT == "off":
    ENR_IDS, V_ENR = [], np.zeros((0, V_TAT.shape[1]), dtype="float32")
elif ENRICHMENT_VARIANT == "filtered":
    keep = {r["verse_id"] for r in jl(ROOT/"data"/"enrichment_selective.jsonl")
            if r.get("life_situations_te")}
    sel = [i for i, v in enumerate(_E_IDS) if v in keep]
    ENR_IDS, V_ENR = [_E_IDS[i] for i in sel], _E_ALL[sel]
else:                                    # "full"
    ENR_IDS, V_ENR = _E_IDS, _E_ALL
BM25 = BM25Okapi([bm25_tokens(retrieval_text(r)) for r in CORPUS])
print(f"vectors: tat={V_TAT.shape} enr={V_ENR.shape} variant={ENRICHMENT_VARIANT}")
print("ready.")

def rrf(lists, k=60):
    out = {}
    for lst in lists:
        for i, d in enumerate(lst): out[d] = out.get(d, 0.0) + 1.0/(k+i+1)
    return out

@spaces.GPU(duration=60)
def search(query, top_k=8, pool=RERANK_POOL):
    """GPU is allocated only for this call - query embedding + reranking."""
    BI.to("cuda"); CE.model.to("cuda")
    q = np.asarray(BI.encode([query], normalize_embeddings=True))[0]
    sims = {IDS[j]: float(V_TAT[j] @ q) for j in range(len(IDS))}
    if len(ENR_IDS):                      # max-pool over the two surfaces
        es = V_ENR @ q
        for j, vid in enumerate(ENR_IDS): sims[vid] = max(sims[vid], float(es[j]))
    dense = [d for d, _ in sorted(sims.items(), key=lambda kv: -kv[1])[:50]]
    sc  = BM25.get_scores(bm25_tokens(query))
    lex = [IDS[j] for j in np.argsort(-sc)[:50]]
    ranked = [d for d, _ in sorted(rrf([dense, lex]).items(), key=lambda kv: -kv[1])]
    cand = ranked[:pool]
    s = np.asarray(CE.predict([(query, BY_ID[c].get("tatparyam_te","")) for c in cand], batch_size=64))
    if len(ENR_IDS):
        idx = [i for i, c in enumerate(cand) if c in ENR and enrichment_text(ENR[c])]
        if idx:
            s2 = CE.predict([(query, enrichment_text(ENR[cand[i]])) for i in idx], batch_size=64)
            for k, i in enumerate(idx): s[i] = max(s[i], float(s2[k]))
    return [BY_ID[cand[i]] for i in np.argsort(-s)[:top_k]]

# ---------------------------------------------------------------- generation
class Route(BaseModel):
    kind: Literal["lookup","thematic","out_of_scope"]

class Grade(BaseModel):
    sufficient: bool
    reason: str

class Citation(BaseModel):
    verse_id: str = Field(description="exact id from the passages, e.g. gita-2-47")
    quote_te: str = Field(description="VERBATIM substring copied from that passage's tatparyam")
    claim_te: str

class Answer(BaseModel):
    answer_te: str
    citations: list[Citation]

def _client():
    from anthropic import Anthropic
    return Anthropic()

def ask(schema, system, user, model=MODEL_FAST, think=0):
    kw = {"thinking": {"type":"enabled","budget_tokens":think}} if think else {}
    return _client().messages.parse(model=model, max_tokens=1500, system=system,
        messages=[{"role":"user","content":user}], output_format=schema, **kw).parsed_output

GRADE_LOOKUP = ("Judge whether the supplied Gita passages genuinely answer this factual "
    "question. Be strict. Answering from your own knowledge of the Gita is a failure - "
    "only the passages count.")
GRADE_THEMATIC = ("Someone described a situation in their life. Judge whether the passages "
    "speak to it. The standard is what a pravachanam would use, NOT an advice column: the "
    "Gita gives principles, not instructions. A passage is sufficient if the tradition would "
    "reasonably offer it to someone in this situation. Say insufficient only when the "
    "passages are about something genuinely different. Only the passages count.")
GEN_SYSTEM = """You answer using ONLY the Bhagavad Gita passages provided.
- Answer in modern, natural Telugu (వ్యావహారిక).
- You know the Gita well; that knowledge must not enter the answer.
- Never invent a verse, a number, or a quotation.
- Every substantive claim needs a citation whose quote_te is copied VERBATIM as a
  contiguous substring of that passage's తాత్పర్యం.
- Where passages differ, say so. Offer what the text says; do not instruct the reader.
- Be concise: 3-5 sentences. Telugu output tokens are slow; length is not quality."""

REFUSAL = ("ఈ ప్రశ్నకు సమాధానం భగవద్గీతలో నాకు దొరకలేదు. "
           "ఈ వ్యవస్థ భగవద్గీత (700 శ్లోకాలు) నుండి మాత్రమే సమాధానం ఇస్తుంది.")

def passages(ctx):
    return "\n\n".join(f"[{r['id']}] {r['citation_te']}\nతాత్పర్యం: {r['tatparyam_te']}" for r in ctx)

STREAM_SYSTEM = """You answer using ONLY the Bhagavad Gita passages provided.

If the passages do not speak to the question, reply with exactly:
INSUFFICIENT
and nothing else.

Otherwise reply in EXACTLY this format:

ANSWER
<your answer in modern natural Telugu (వ్యావహారిక), 3-5 sentences>
CITATIONS
<verse_id> | <quote copied VERBATIM from that passage's తాత్పర్యం>
<verse_id> | <quote copied VERBATIM>

THE HARD RULE: every sentence of the answer must be something a reader could
verify by reading the passages alone. You know the Gita well. That knowledge is
the main risk here - it makes you add things that are true of the text but
absent from THESE passages. Do not.

Four specific failures to avoid:

1. Do not swap in your own vocabulary. If a passage says "నా ప్రకృతి", write
   "నా ప్రకృతి" - not "ఈశ్వరుడు". If it says "బ్రహ్మస్వరూపమును పొందుటకు
   అర్హుడు", do not compress that to "మోక్షం". Use the passage's own words for
   its own concepts.
2. Do not attribute a claim to a verse that does not make it. If a verse is
   about తపస్సు, do not report it as being about త్యాగం.
3. Do not state a conclusion the passages leave unstated, however sound it
   seems. If no passage compares the self to the body, do not introduce that
   comparison.
4. Do not drop conditions. If a passage makes something conditional on several
   things, do not present it as following from one.

Also:
- Never invent a verse, a number, or a quotation.
- Each quote must be a contiguous substring of that passage's తాత్పర్యం.
- Where passages differ, say so. Offer what the text says; do not instruct the reader.
- Prefer a shorter answer that stays inside the passages over a fuller one that
  steps outside them. Saying less is the correct behaviour, not a weakness."""

def anchor_span(quote: str, source: str, min_run: int = 24):
    """Confirm the model was pointing at real source text, then return the SOURCE
    span - never the model's rendering.

    Exact-substring matching rejected citations over a single dropped character
    ("ఫలాలపై" vs "ఫలాలపైన"). Instead we require a long contiguous run of the
    model's quote to occur verbatim in the source, and display the source text
    around it. Nothing shown to the user originates from the model, so the
    guarantee is unchanged while transcription drift is tolerated.
    """
    q, src = squash(quote), squash(source)
    if not q or not src: return None
    best = 0; pos = -1
    for i in range(len(q)):
        for j in range(len(q), i + best, -1):
            if q[i:j] in src:
                if j - i > best: best, pos = j - i, src.index(q[i:j])
                break
    if best < min_run: return None
    # map back to the unsquashed source: walk until we've seen `pos` non-space chars
    seen = 0; start = 0
    for k, ch in enumerate(source):
        if not ch.isspace():
            if seen == pos: start = k; break
            seen += 1
    tail = source[start:]
    return (tail[:240].rstrip() + ("…" if len(tail) > 240 else "")) or None

def answer(question: str):
    """Streaming generator: yields (answer_so_far, citations_markdown)."""
    import time
    from concurrent.futures import ThreadPoolExecutor
    if not question.strip():
        yield "", ""; return
    t0 = time.time(); q = norm(question)

    if LATIN.search(q) and not TELUGU.search(q):
        class Translit(BaseModel): telugu: str
        q = ask(Translit, "Transliterate romanized Telugu (Tenglish) into Telugu "
                          "script. If the input is English, translate to Telugu.", q).telugu

    # route (API-bound) and retrieval (GPU-bound) are independent - run together
    with ThreadPoolExecutor(2) as ex:
        f_route = ex.submit(ask, Route,
            "Route questions for a Bhagavad Gita retrieval system. The corpus is the "
            "Gita's 700 verses only. Ramayana, Mahabharata narrative or unrelated "
            "topics are out_of_scope.", q)
        f_ctx = ex.submit(search, q)
        route = f_route.result().kind
        ctx = f_ctx.result()
    t_ready = time.time() - t0
    if route == "out_of_scope":
        yield REFUSAL, ""; return

    # one streaming call; no separate grade round trip - INSUFFICIENT is the refusal
    body = f"ప్రశ్న: {q}\n\nPASSAGES:\n{passages(ctx)}"
    buf, first = "", None
    with _client().messages.stream(model=MODEL_GEN, max_tokens=3000,
                                   system=STREAM_SYSTEM,
                                   messages=[{"role":"user","content":body}]) as stream:
        for tok in stream.text_stream:
            buf += tok
            if first is None and buf.strip(): first = time.time() - t0
            if buf.lstrip().startswith("INSUF"): continue
            shown = buf.split("CITATIONS")[0].replace("ANSWER", "", 1).strip()
            if shown: yield shown, ""

    if buf.strip().startswith("INSUFFICIENT"):
        yield REFUSAL, ""; return

    _has = "CITATIONS" in buf
    _lines = [l for l in buf.split("CITATIONS")[-1].splitlines() if "|" in l]
    print(f"CITEDEBUG | has_block={_has} pipe_lines={len(_lines)} "
          f"buf_chars={len(buf)} tail={buf[-120:]!r}", flush=True)
    FOREIGN = re.compile(r"[\u00C0-\u024F\u0400-\u04FF\u0900-\u097F\u0980-\u09FF"
                         r"\u0B80-\u0BFF\u0C80-\u0CFF]")
    _bad = FOREIGN.findall(buf.split("CITATIONS")[0])
    if _bad: print(f"SCRIPTWARN | {len(_bad)} foreign chars {''.join(sorted(set(_bad)))[:12]!r}", flush=True)
    ans = buf.split("CITATIONS")[0].replace("ANSWER", "", 1).strip()
    by = {r["id"]: r for r in ctx}
    ok = []
    for line in buf.split("CITATIONS")[-1].splitlines():
        if "|" not in line: continue
        vid, _, quote = line.partition("|")
        # the model copies the id as it appears in the passages, often keeping the
        # brackets ("[gita-4-20]") or adding markdown. Pull the bare id out.
        # the model writes the id loosely: "[gita-2-47]", "gita-2-47", "2-47",
        # sometimes "2.47". Accept any of those and normalise.
        m = re.search(r"(?:gita[-\s]*)?(\d{1,2})[-.](\d{1,3})", vid)
        rec = by.get(f"gita-{m.group(1)}-{m.group(2)}") if m else None
        if not rec:
            print(f"CITEDEBUG | unmatched id {vid.strip()[:40]!r} "
                  f"(ctx has {list(by)[:3]}...)", flush=True)
            continue
        span = anchor_span(quote.strip(), rec["tatparyam_te"])
        if span: ok.append((span, rec))
    cites = "\n\n".join(
        f"**{rec['citation_te']}**  \n> {qt}  \n[మూలం]({rec['source_url']})"
        + ("" if rec.get("verse_number_verified", True) else "  \n_(శ్లోక సంఖ్య ధృవీకరించబడలేదు)_")
        for qt, rec in ok)
    total = time.time() - t0
    print(f"TIMING | model={MODEL_GEN} ready {t_ready:.1f}s · first_token {first or 0:.1f}s · "
          f"total {total:.1f}s · cited {len(ok)}", flush=True)
    yield ans, (cites or "_(ధృవీకరించిన ఉల్లేఖనాలు లేవు)_") + \
          f"\n\n<sub>first word {first or 0:.1f}s · total {total:.1f}s</sub>"

EXAMPLES = ["కర్మ ఫలం గురించి కృష్ణుడు ఏమి చెప్పాడు?",
            "నా ఉద్యోగంలో నాకు సరైన గుర్తింపు రావడం లేదు, ఏం చేయాలి?",
            "kopam ela puttundi gita lo",
            "ఆత్మ చనిపోతుందా?"]

with gr.Blocks(title="భగవద్గీత — ప్రశ్నోత్తరాలు") as demo:
    gr.Markdown("# భగవద్గీత — ప్రశ్నోత్తరాలు\n"
                "భగవద్గీత (699 శ్లోకాలు) నుండి మాత్రమే సమాధానాలు. "
                "ప్రతి ఉల్లేఖనం మూల శ్లోకంతో సరిపోల్చి ధృవీకరించబడుతుంది.\n\n"
                "*సమాధానాలు యంత్రం రూపొందించినవి. మూలాన్ని చూసి నిర్ధారించుకోండి.*")
    inp = gr.Textbox(label="ప్రశ్న", placeholder="తెలుగులో లేదా Tenglish లో అడగండి...", lines=2)
    btn = gr.Button("అడగండి", variant="primary")
    out = gr.Markdown(label="సమాధానం")
    cit = gr.Markdown(label="ఆధారాలు")
    btn.click(answer, inputs=inp, outputs=[out, cit])
    inp.submit(answer, inputs=inp, outputs=[out, cit])
    gr.Examples(EXAMPLES, inputs=inp)

if __name__ == "__main__":
    demo.launch()
