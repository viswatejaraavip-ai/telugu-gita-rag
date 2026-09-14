"""Judge the DEPLOYED Space, not the local pipeline.

Collect: query the Space API. Judge: score with a stronger model than the one
that generated (Opus grading Haiku).

Scope note: the Space returns (answer, citations) but not the full retrieved
set, so the judge grades against the CITED passages. That covers faithfulness,
relevance and register; it cannot score whether apt passages went unused.
"""
from __future__ import annotations
import json, re, sys, time
from pathlib import Path
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
SPACE = "https://amrithatejaswiservices-telugu-gita-rag.hf.space"
JUDGE_MODEL = "claude-opus-5"
RUNS  = ROOT/"eval"/"space_runs.jsonl"
SCORE = ROOT/"eval"/"space_judgments.jsonl"

class Judgment(BaseModel):
    faithful: int = Field(ge=1, le=5, description="5 = every statement traceable to the cited passages; 1 = major unsupported claims")
    relevant: int = Field(ge=1, le=5, description="5 = directly addresses the question; 1 = answers something else")
    register: int = Field(ge=1, le=5, description="5 = natural modern Telugu, offers what the text says; 1 = broken Telugu or instructs the reader like a guru")
    unsupported_claims: list[str] = Field(default_factory=list)
    reasoning: str

JUDGE_SYS = """You grade a Bhagavad Gita question-answering system.

You get a question, the Telugu passages the system CITED, and its answer. Grade
only against those passages.

You likely know the Gita. Ignore that. A statement true of the Gita but absent
from these passages is UNSUPPORTED - preventing exactly that is why this system
exists. Do not reward length."""

def ask_space(q, timeout=240):
    import requests
    t=time.time()
    r=requests.post(f"{SPACE}/gradio_api/call/answer", json={"data":[q]}, timeout=30)
    eid=r.json()["event_id"]
    res=requests.get(f"{SPACE}/gradio_api/call/answer/{eid}", stream=True, timeout=timeout)
    last=None; first=None
    for raw in res.iter_lines():
        if not raw: continue
        s=raw.decode("utf-8","ignore")
        if s.startswith("data: "):
            try: p=json.loads(s[6:])
            except Exception: continue
            if isinstance(p,list) and p and str(p[0]).strip():
                if first is None: first=time.time()-t
                last=p
    ans = str(last[0]) if last else ""
    cit = str(last[1]) if last and len(last)>1 else ""
    quotes = re.findall(r"\*\*(.+?)\*\*\s*\n>\s*(.+?)\s*\n", cit)
    return dict(answer=ans, citations=[{"citation":c,"quote":q2} for c,q2 in quotes],
                n_cited=len(quotes), first=first, total=time.time()-t)

def collect(n=21):
    src=[json.loads(l) for l in (ROOT/"finetune"/"held_queries.jsonl").open(encoding="utf-8") if l.strip()]
    import random; random.Random(23).shuffle(src)
    per=max(1,n//3); picked=[]; seen={}
    for r in src:
        k=r["kind"]
        if seen.get(k,0)<per: picked.append(r); seen[k]=seen.get(k,0)+1
        if len(picked)>=n: break
    done=set()
    if RUNS.exists(): done={json.loads(l)["question"] for l in RUNS.open(encoding="utf-8") if l.strip()}
    RUNS.parent.mkdir(exist_ok=True)
    todo=[r for r in picked if r["query"] not in done]
    # fire all queries concurrently - each is ~7s of mostly-waiting, so serial
    # collection wasted minutes. ZeroGPU serialises the GPU section itself.
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    lock=threading.Lock(); t0=time.time()
    def one(r):
        try: return r, ask_space(r["query"]), None
        except Exception as e: return r, None, e
    with RUNS.open("a",encoding="utf-8") as f, ThreadPoolExecutor(len(todo) or 1) as ex:
        futs=[ex.submit(one,r) for r in todo]
        for n,fu in enumerate(as_completed(futs),1):
            r,out,err=fu.result()
            if err:
                print(f"  {n}/{len(todo)} FAILED {type(err).__name__} {str(err)[:60]}", flush=True); continue
            with lock:
                f.write(json.dumps(dict(question=r["query"], kind=r["kind"], gold=r["verse_id"], **out),
                                   ensure_ascii=False)+"\n"); f.flush()
            print(f"  {n}/{len(todo)} [{r['kind']:<8}] cited={out['n_cited']} "
                  f"first={out['first'] or 0:.1f}s total={out['total']:.1f}s", flush=True)
    print(f"  wall {time.time()-t0:.0f}s for {len(todo)} queries", flush=True)

def judge():
    from anthropic import Anthropic
    cl=Anthropic()
    # The model saw the FULL tatparyam of each retrieved verse; anchor_span only
    # returns the span from the match point on. Judging against the excerpt
    # penalised answers grounded in text the judge could not see (e.g. 6.17 does
    # name food and work, but the quoted span started after that). Resolve each
    # citation label back to the full verse.
    corpus=[json.loads(l) for l in (ROOT/"corpus"/"gita.jsonl").open(encoding="utf-8") if l.strip()]
    full={}
    for r in corpus:
        full[f"{r['adhyaya_no']}.{r['sloka_no']}"]=r['tatparyam_te']
    def expand(c):
        m=re.search(r"(\d{1,2})\.(\d{1,3})", c["citation"])
        t=full.get(f"{m.group(1)}.{m.group(2)}") if m else None
        return f"{c['citation']}\n{t or c['quote']}"
    runs=[json.loads(l) for l in RUNS.open(encoding="utf-8") if l.strip()]
    done=set()
    if SCORE.exists(): done={json.loads(l)["question"] for l in SCORE.open(encoding="utf-8") if l.strip()}
    todo=[r for r in runs if r["question"] not in done and r["answer"].strip()]
    print(f"judging {len(todo)} answers with {JUDGE_MODEL}", flush=True)
    with SCORE.open("a",encoding="utf-8") as f:
        for i,r in enumerate(todo,1):
            cites="\n\n".join(expand(c) for c in r["citations"]) or "(no citations)"
            body=f"QUESTION:\n{r['question']}\n\nCITED PASSAGES:\n{cites}\n\nANSWER:\n{r['answer']}"
            j=cl.messages.parse(model=JUDGE_MODEL, max_tokens=3000, system=JUDGE_SYS,
                                messages=[{"role":"user","content":body}],
                                output_format=Judgment,
                                output_config={"effort":"medium"},
                                thinking={"type":"adaptive"}).parsed_output
            f.write(json.dumps(dict(question=r["question"],kind=r["kind"],
                                    n_cited=r["n_cited"],**j.model_dump()),ensure_ascii=False)+"\n"); f.flush()
            print(f"  {i}/{len(todo)} [{r['kind']:<8}] faith={j.faithful} rel={j.relevant} reg={j.register}", flush=True)

def report():
    import statistics as st
    from collections import defaultdict
    rows=[json.loads(l) for l in SCORE.open(encoding="utf-8") if l.strip()]
    if not rows: print("no judgments"); return
    dims=["faithful","relevant","register"]
    print(f"\nANSWER CORRECTNESS — deployed Space, n={len(rows)}")
    print("  " + "  ".join(f"{d}={st.mean(r[d] for r in rows):.2f}/5" for d in dims))
    print(f"  mean citations/answer = {st.mean(r['n_cited'] for r in rows):.1f}")
    per=defaultdict(list)
    for r in rows: per[r["kind"]].append(r)
    for k,v in sorted(per.items()):
        print(f"    {k:<9} n={len(v):<3} " + " ".join(f"{d}={st.mean(x[d] for x in v):.2f}" for d in dims)
              + f"  cites={st.mean(x['n_cited'] for x in v):.1f}")
    bad=[r for r in rows if r["unsupported_claims"]]
    print(f"\n  answers with unsupported claims: {len(bad)}/{len(rows)}")
    for r in bad[:4]: print(f"    [{r['kind']}] {r['unsupported_claims'][0][:90]}")
    low=[r for r in rows if min(r[d] for d in dims)<=2]
    if low:
        print(f"\n  weak answers (any dim <=2): {len(low)}")
        for r in low[:3]: print(f"    [{r['kind']}] {r['reasoning'][:110]}")

if __name__=="__main__":
    c=sys.argv[1] if len(sys.argv)>1 else "report"
    if c=="collect": collect(int(sys.argv[2]) if len(sys.argv)>2 else 21)
    elif c=="judge": judge()
    else: report()
