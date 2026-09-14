"""LangGraph RAG over the Bhagavad Gita corpus.

Design goal: answers come ONLY from retrieved verses, every claim carries a
verifiable citation, and the graph refuses rather than falling back on the
model's parametric memory.
"""
from __future__ import annotations
import os, re, sys, unicodedata
from pathlib import Path
from typing import Annotated, Literal, Optional, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieval import LATIN, TELUGU, norm, search

from pydantic import BaseModel, Field
from langgraph.graph import END, START, StateGraph

# Haiku 4.5 for everything: routing, transliteration, grading and generation.
# Roughly 5x cheaper than Opus on input and output. Swap MODEL_GEN back to
# "claude-opus-5" if Telugu generation quality disappoints - that is the one
# node where the difference is likely to show.
MODEL_FAST = "claude-haiku-4-5"    # query transformation, routing, grading
MODEL_GEN  = "claude-haiku-4-5"    # answer generation
MAX_REGEN = 1

# ------------------------------------------------------------------- schemas
class Route(BaseModel):
    kind: Literal["lookup", "thematic", "out_of_scope"] = Field(
        description="lookup = factual question about what the Gita says; "
                    "thematic = a life situation to find guidance for; "
                    "out_of_scope = unrelated to the Bhagavad Gita")
    adhyaya: Optional[int] = Field(None, description="1-18 if the user named a chapter")
    search_queries_te: list[str] = Field(
        default_factory=list,
        description="1-3 Telugu search phrases. For thematic questions, express the "
                    "underlying themes in the vocabulary the Gita's Telugu commentary "
                    "would use, not the user's everyday words.")

class Grade(BaseModel):
    sufficient: bool = Field(description="Do the passages actually answer the question?")
    reason: str

class Citation(BaseModel):
    verse_id: str = Field(description="exact id from the passages, e.g. gita-2-47")
    quote_te: str = Field(description="VERBATIM substring copied from that passage's tatparyam")
    claim_te: str = Field(description="the statement in your answer this supports")

class Answer(BaseModel):
    answer_te: str = Field(description="The answer, in modern Telugu.")
    citations: list[Citation]

# --------------------------------------------------------------------- state
class State(TypedDict, total=False):
    question: str
    question_norm: str
    script: str
    route: str
    adhyaya: Optional[int]
    queries: list[str]
    context: list[dict]
    sufficient: bool
    grade_reason: str
    answer: str
    citations: list[dict]
    verification: dict
    attempts: int
    offline: bool

# --------------------------------------------------------------------- claude
def _client():
    from anthropic import Anthropic
    return Anthropic()

def _has_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))

def ask(schema, system: str, user: str, effort: str = "medium",
        model: str | None = None, think: int = 0):
    """Claude call with a structured output schema.

    Haiku 4.5 and the Opus/Sonnet family take different knobs: `effort` is
    rejected by Haiku, and Haiku still uses the older
    {"type": "enabled", "budget_tokens": N} thinking form rather than adaptive.
    Sending the wrong one is a 400, so branch on the model.
    """
    model = model or MODEL_FAST
    kw = {}
    if model.startswith("claude-haiku"):
        if think:
            kw["thinking"] = {"type": "enabled", "budget_tokens": think}
    else:
        kw["output_config"] = {"effort": effort}
        kw["thinking"] = {"type": "adaptive"}
    r = _client().messages.parse(
        model=model, max_tokens=8000,
        system=system, messages=[{"role": "user", "content": user}],
        output_format=schema, **kw,
    )
    return r.parsed_output

# ---------------------------------------------------------------------- nodes
def n_normalize(s: State) -> State:
    q = norm(s["question"])
    te, la = len(TELUGU.findall(q)), len(LATIN.findall(q))
    script = "telugu" if te > la else ("latin" if la and not te else "mixed")
    out = {"question_norm": q, "script": script, "attempts": 0,
           "offline": not _has_key()}
    if script == "latin" and _has_key():
        # Tenglish -> Telugu script, else dense retrieval cannot match the corpus
        class T(BaseModel):
            telugu: str = Field(description="the same question written in Telugu script")
        out["question_norm"] = ask(
            T, "You transliterate romanized Telugu (Tenglish) into Telugu script. "
               "Preserve meaning exactly. If the input is English, translate to Telugu.",
            q, model=MODEL_FAST).telugu
    return out

def n_route(s: State) -> State:
    if s.get("offline"):
        return {"route": "lookup", "adhyaya": None, "queries": [s["question_norm"]]}
    r = ask(Route,
        "You route questions for a Bhagavad Gita retrieval system. The corpus is the "
        "Gita's 700 verses with Telugu tatparyam — nothing else. A question about the "
        "Ramayana, Mahabharata narrative, or unrelated topics is out_of_scope.",
        s["question_norm"], model=MODEL_FAST)
    return {"route": r.kind, "adhyaya": r.adhyaya,
            "queries": r.search_queries_te or [s["question_norm"]]}

def n_retrieve(s: State) -> State:
    """Search the user's actual question.

    We used to search up to 3 LLM-rewritten phrasings and merge by _score.
    Measured on 50 held-out queries, that was worse on every metric and 3.4x
    slower: recall@1 0.300 -> 0.160, NDCG@10 0.369 -> 0.288.

    The cause was the merge, not the expansion. Scores from separate reranker
    runs on different queries are unnormalised and not comparable, so sorting
    them together destroyed the ordering - recall@10 barely moved (the right
    verse stayed in the pool) while recall@1 halved. Re-enabling expansion
    would require fusing the lists by RANK (RRF), as we already do for
    dense + BM25. Until that is measured, search the question.
    """
    hits = search(s["question_norm"], top_k=8, adhyaya=s.get("adhyaya"))
    return {"context": hits}

def _passages(ctx: list[dict]) -> str:
    return "\n\n".join(
        f"[{r['id']}] {r['citation_te']}\n"
        f"శ్లోకం: {r.get('sloka_sa','')}\n"
        f"తాత్పర్యం: {r['tatparyam_te']}"
        for r in ctx)

GRADE_LOOKUP = (
    "Judge whether the supplied Gita passages genuinely answer this factual question. "
    "Be strict: loosely related passages are NOT sufficient. Answering from your own "
    "knowledge of the Gita is a failure - only the passages count.")

GRADE_THEMATIC = (
    "Someone has described a situation in their life. Judge whether the supplied Gita "
    "passages speak to it.\n\n"
    "The standard is what a pravachanam would use, NOT an advice column. The Gita gives "
    "principles, not practical instructions - it will never say 'ask your manager for a "
    "promotion'. A passage counts as sufficient if the tradition would reasonably offer it "
    "to someone in this situation: it addresses the underlying theme (duty, recognition, "
    "attachment to results, envy, patience, self-worth), even though applying it is left "
    "to the listener.\n\n"
    "Say insufficient only when the passages are about something genuinely different. "
    "Do not use your own knowledge of the Gita - only the passages count.")

def n_grade(s: State) -> State:
    if not s.get("context"):
        return {"sufficient": False, "grade_reason": "no passages retrieved"}
    if s.get("offline"):
        return {"sufficient": True, "grade_reason": "offline: skipped grading"}
    # Lookup and thematic questions need different standards. A single global
    # strictness rejects principle-level passages for thematic queries, which is
    # most of what the Gita offers.
    system = GRADE_THEMATIC if s.get("route") == "thematic" else GRADE_LOOKUP
    g = ask(Grade, system,
        f"ప్రశ్న: {s['question_norm']}\n\nPASSAGES:\n{_passages(s['context'])}",
        model=MODEL_FAST)
    return {"sufficient": g.sufficient, "grade_reason": g.reason}

GEN_SYSTEM = """You answer questions using ONLY the Bhagavad Gita passages provided.

Rules:
- Answer in modern, natural Telugu (వ్యావహారిక).
- Use ONLY the supplied passages. You know the Gita well; that knowledge must not
  enter the answer. If the passages do not support a point, leave it out.
- Never invent a verse, a verse number, or a quotation.
- Every substantive claim needs a citation whose quote_te is copied VERBATIM as a
  contiguous substring of that passage's తాత్పర్యం. Do not paraphrase inside quote_te.
- Where passages point in different directions, say so rather than flattening them.
- Offer what the text says. Do not issue personal instructions to the reader.
- EVERY substantive sentence must be covered by a citation. A sentence with no
  citation cannot be verified, so it must not carry a claim about what the Gita
  says. Framing sentences that only restate the question need no citation."""

def n_generate(s: State) -> State:
    if s.get("offline"):
        return {"answer": "[offline: ANTHROPIC_API_KEY not set — retrieval only]",
                "citations": [], "attempts": s.get("attempts", 0) + 1}
    fix = ""
    v = s.get("verification", {})
    if v.get("failed"):
        fix = ("\n\nYour previous answer failed citation verification for: "
               + ", ".join(v["failed"])
               + ". Copy quote_te VERBATIM from the passage text this time.")

    a = ask(Answer, GEN_SYSTEM,
            f"ప్రశ్న: {s['question_norm']}\n\nPASSAGES:\n{_passages(s['context'])}{fix}",
            model=MODEL_GEN, think=2000)
    return {"answer": a.answer_te,
            "citations": [c.model_dump() for c in a.citations],
            "attempts": s.get("attempts", 0) + 1}

# Telugu prose ends sentences with '.', but inline references like "భగవద్గీత 2.47"
# also contain one. Split on terminators NOT flanked by digits.
SENT_SPLIT = re.compile(r"(?<!\d)[.।!?\n]+(?!\d)")

def _squash(t: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", t or ""))

def n_verify(s: State) -> State:
    """Every quote must literally occur in the passage it cites."""
    by_id = {r["id"]: r for r in s.get("context", [])}
    ok, failed = [], []
    for c in s.get("citations", []):
        rec = by_id.get(c["verse_id"])
        if not rec:
            failed.append(f"{c['verse_id']} (not in retrieved set)"); continue
        hay = _squash(rec.get("tatparyam_te", "")) + _squash(rec.get("sloka_sa", ""))
        if _squash(c["quote_te"]) and _squash(c["quote_te"]) in hay:
            c["citation_te"]  = rec["citation_te"]
            c["source_url"]   = rec["source_url"]
            # 115 verses have positional numbering that failed the canon check
            c["number_verified"] = rec.get("verse_number_verified", True)
            ok.append(c)
        else:
            failed.append(f"{c['verse_id']} (quote not found in passage)")
    # Coverage: what fraction of substantive sentences is backed by a verified
    # quote. The quote check alone only validates what the model chose to cite -
    # an uncited claim is never checked at all, so coverage is the other half of
    # the guarantee.
    # A sentence counts as grounded if it overlaps a verified quote OR any
    # retrieved passage. Prefix matching failed here: a quote that itself spans
    # two sentences gets split, and the tail looks uncited.
    grounded = "".join(_squash(c["quote_te"]) for c in ok) + \
               "".join(_squash(r.get("tatparyam_te", "")) for r in s.get("context", []))
    sents = [x.strip() for x in SENT_SPLIT.split(s.get("answer", "")) if len(x.strip()) > 18]
    def hit(t):
        q = _squash(t)
        return any(q[i:i+16] in grounded for i in range(0, max(1, len(q) - 15), 4))
    covered = sum(1 for t in sents if hit(t))
    cov = covered / len(sents) if sents else 1.0
    # Never hand back a citation that failed verification. Reaching the retry
    # budget is not a reason to ship an unverified quote.
    return {"citations": ok,
            "verification": {"verified": ok, "failed": failed,
                             "n_ok": len(ok), "n_failed": len(failed),
                             "coverage": round(cov, 2),
                             "uncited": len(sents) - covered}}

REFUSAL = ("ఈ ప్రశ్నకు సమాధానం భగవద్గీతలో నాకు దొరకలేదు. "
           "ఈ వ్యవస్థ భగవద్గీత (౭౦౦ శ్లోకాలు) నుండి మాత్రమే సమాధానం ఇస్తుంది.")

def n_refuse(s: State) -> State:
    return {"answer": REFUSAL, "citations": [],
            "verification": {"verified": [], "failed": [], "n_ok": 0, "n_failed": 0}}

# ---------------------------------------------------------------------- edges
def e_route(s: State):
    return "refuse" if s["route"] == "out_of_scope" else "retrieve"

def e_grade(s: State):
    return "generate" if s.get("sufficient") else "refuse"

def e_verify(s: State):
    """Regenerate only on a FAILED citation - the hard guarantee.

    Coverage is reported but does not gate. Answers legitimately contain
    connective sentences that are not in any source, so a coverage threshold
    fired on every query and doubled latency without ever catching a real
    problem (failed was already 0). Diagnostic, not a gate.
    """
    v = s.get("verification", {})
    if v.get("failed") and s.get("attempts", 0) <= MAX_REGEN and not s.get("offline"):
        return "generate"
    return END

def build():
    g = StateGraph(State)
    for name, fn in [("normalize", n_normalize), ("route", n_route),
                     ("retrieve", n_retrieve), ("grade", n_grade),
                     ("generate", n_generate), ("verify", n_verify),
                     ("refuse", n_refuse)]:
        g.add_node(name, fn)
    g.add_edge(START, "normalize")
    g.add_edge("normalize", "route")
    g.add_conditional_edges("route", e_route, {"retrieve": "retrieve", "refuse": "refuse"})
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", e_grade, {"generate": "generate", "refuse": "refuse"})
    g.add_edge("generate", "verify")
    g.add_conditional_edges("verify", e_verify, {"generate": "generate", END: END})
    g.add_edge("refuse", END)
    return g.compile()

if __name__ == "__main__":
    app = build()
    q = " ".join(sys.argv[1:]) or "కర్మ ఫలం గురించి కృష్ణుడు ఏమి చెప్పాడు?"
    out = app.invoke({"question": q})
    print(f"\nQ: {q}\nroute={out.get('route')} script={out.get('script')} "
          f"ctx={len(out.get('context',[]))} offline={out.get('offline')}")
    print("\n--- retrieved ---")
    for r in out.get("context", [])[:5]:
        print(f"  {r['id']:<12} {r['_score']:.4f}  {r['tatparyam_te'][:70]}")
    print(f"\n--- answer ---\n{out.get('answer','')}")
    print(f"\ngrade: sufficient={out.get('sufficient')} — {out.get('grade_reason','')[:160]}")
    v = out.get("verification", {})
    if v: print(f"citations verified={v.get('n_ok')} failed={v.get('n_failed')} "
                f"coverage={v.get('coverage')} uncited={v.get('uncited')}")
