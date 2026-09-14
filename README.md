# Telugu Bhagavad Gita RAG

Grounded question answering over the Bhagavad Gita in Telugu. Every quotation is
verified **verbatim against the source verse** before it reaches the user, and the
system refuses rather than answering from the model's own memory.

**[Live demo →](https://huggingface.co/spaces/amrithatejaswiservices/telugu-gita-rag)**

Ask in Telugu (`కర్మ ఫలం గురించి కృష్ణుడు ఏమి చెప్పాడు?`) or romanized Tenglish
(`kopam ela puttundi gita lo`). Ask about the Ramayana and it declines.

---

## Why this exists

General assistants answer questions about the Gita from parametric memory: no
citations, no way to check, and claims that blend the text with everything else
the model absorbed. This system answers **only** from a 699-verse corpus and
proves it.

## Results

### Retrieval — 320 held-out queries

| stage | recall@10 | NDCG@10 |
|---|---|---|
| stock BGE-M3 | 0.350 | 0.256 |
| + fine-tuned bi-encoder | 0.512 | 0.358 |
| + fine-tuned cross-encoder | **0.634** | **0.468** |

**+81% recall, +83% NDCG.** By query type:

| kind | recall@10 | NDCG@10 | median rank |
|---|---|---|---|
| lookup | 0.758 → **0.883** | 0.582 → **0.714** | 2 → **1** |
| romanized (tenglish) | 0.359 → **0.641** | 0.256 → **0.483** | 7 → **1** |
| thematic | 0.344 → 0.383 | 0.186 → 0.214 | 12 → 10 |

Romanized Telugu gained most. A cross-encoder sees `kopam` and `కోపం` in one
forward pass and aligns them directly; a bi-encoder comparing two independently
computed vectors cannot.

### Answer quality — 21 held-out queries, judged by Claude Opus 5

| | Haiku 4.5 | **Sonnet 5** |
|---|---|---|
| faithfulness | 3.86/5 | **4.76/5** |
| relevance | 4.43/5 | **4.67/5** |
| register (Telugu quality) | 4.05/5 | **4.81/5** |
| answers with unsupported claims | 76% | **33%** |
| foreign-script corruption | 1/21 | **0/21** |
| citations verified | 100% | 100% |

## How this was built

The interesting part wasn't the RAG plumbing — it was finding out which
assumptions were wrong, and in what order.

### 1. The corpus barely exists

The plan was Ramayana, Mahabharata and Gita. An audit of te.wikisource.org killed
two thirds of it: **10,842 Ramayana slokas, of which 64 have a Telugu tatparyam —
0.6%.** The rest is Sanskrit mechanically transliterated into Telugu script, which
*looks* like Telugu to a character counter and isn't. The Mahabharata is the same,
and it's Vyasa's Sanskrit rather than Nannaya's Telugu Andhra Mahabharatam.

Only the Gita had genuine Sanskrit + Telugu alignment. Scope collapsed to 699
verses on day one. Parsing them needed three different formats — full sloka with
`|| c-v ||` refs, incipit-only, and a numbered variant — because the source isn't
uniform. 16 of 18 chapters then matched canonical verse counts exactly, and the two
that didn't were a duplicated verse and a missing one, not parser bugs.

### 2. The problem wasn't the language, it was the question shape

First baseline: **recall@10 = 0.57** on hand-written questions. The obvious story
was "multilingual embedders are weak on Telugu."

That story was wrong, and testing it was the highest-value hour of the project. A
*floor test* — query with a verbatim 45-character slice of a verse and require the
encoder to rank it first — scored **26/30 at rank 1, zero misses in the top 20**.
A topic-matched anisotropy check put Telugu at 0.485 against English 0.502. The
encoder reads Telugu fine.

The real failure was **query-document asymmetry**: questions are interrogative,
verses are 126-character declaratives where the answer is one clause among several.
That's the hardest case for a bi-encoder in *any* language — and it's fixable by
fine-tuning, which "the model can't read Telugu" would not have been.

### 3. Training data from the corpus itself

No labelled data exists, so Claude generated it: 5 queries per verse — 2 factual,
2 first-person life situations, 1 romanized Tenglish — over 581 training verses.
**2,905 pairs for $4.71.**

Two rules mattered more than volume. The generator was **forbidden from reusing the
verse's vocabulary**, because a query echoing the commentary teaches lexical overlap
that BM25 already handles; the dense side needs paraphrase. And the split held out
**verses, not queries** — held-out queries drawn from training verses measure
memorisation and produce a large, satisfying, meaningless improvement.

Hard negatives came from the live retriever: for each query, the verses it ranked
highly and got wrong. Random negatives teach nothing — a battle-conch verse is
trivially separable from a karma query.

### 4. Fine-tuning, measured

Contrastive training on a free Colab T4. **recall@10 0.350 → 0.512.** The mechanism
is visible in the margin between a query and its correct verse versus a hard
negative: **+0.000 → +0.345**, achieved almost entirely by pushing negatives down
(0.500 → 0.177) rather than pulling positives up (0.500 → 0.522).

Romanized Telugu gained most of all, later reaching 0.359 → 0.641 with the
cross-encoder — because cross-attention sees `kopam` and `కోపం` in one forward pass
and aligns them, which a bi-encoder comparing two independent vectors cannot.

### 5. The stock reranker made things worse

Adding `bge-reranker-v2-m3` off the shelf *degraded* quality: it improved `lookup`,
the query type closest to generic retrieval, and damaged the two types fine-tuning
had specifically taught. **A reranker only helps if it is better at the task than
the retriever it reorders** — and by then the bi-encoder was the more adapted model.

Fine-tuning the cross-encoder on the same triplets fixed it: **0.512 → 0.634**.

Pool size was then measured rather than guessed, by reranking the top 50 once and
reading off smaller pools by truncation — five data points for the cost of one.
recall@10 saturates at pool=30; the extra 20 candidates buy +0.005 NDCG for 40%
more compute.

### 6. What didn't work

Three substantial efforts produced nothing, and each took a measurement to
establish rather than an argument:

- **Query expansion** dropped recall@1 from 0.300 to 0.160 at 3.4× the latency —
  because results from separate reranker runs were merged by *score*, and
  cross-query scores aren't comparable.
- **The thematic enrichment layer** lifted thematic recall@50 but not recall@10,
  and vanished entirely once the reranker ran.
- **Selective enrichment** removed 55% of held-out thematic gold verses. The skip
  rules excluded "metaphysical description" — not realising that *who is God* and
  *what am I* are thematic questions whose answers are exactly those verses.

### 7. Generation: enforce, don't ask

The generation half took 45 seconds and answered well. Getting it to ~10s with a
3-second first word meant removing a thinking budget that bought nothing, folding
the sufficiency check into generation, running routing and retrieval in parallel,
and streaming.

Then the citation rate silently dropped to zero — twice — because the parser
rejected valid citations over bracket punctuation and a missing `gita-` prefix. The
model's format compliance had been perfect throughout. Both bugs were found by a
`cited N` counter added while chasing a different hypothesis, and both looked
exactly like a model that had chosen not to cite.

The last lesson generalises: **tightening the generation prompt against four
observed drift modes, with concrete Telugu examples, changed nothing (76% → 76%).
Switching Haiku → Sonnet did (76% → 33%).** Verbatim citation verification works at
100% because it *enforces*; the prompt that asked for verbatim quotes never did.

## Architecture

```
question
   ├─ transliterate   romanized Telugu → Telugu script (if needed)
   ├─ route ─────────── out_of_scope → refuse
   │        ‖  (parallel)
   ├─ retrieve ──────── fine-tuned BGE-M3  ⨁  char-4gram BM25  → RRF
   │                    → fine-tuned cross-encoder rerank (pool 30)
   ├─ generate ──────── streamed; INSUFFICIENT → refuse
   └─ verify ────────── every quote anchored verbatim to source, else dropped
```

Three design decisions carry most of the quality:

**Embed the tatparyam, not the Sanskrit.** The Telugu commentary is modern prose
in the same register users type in. Embedding the Sanskrit would put query and
document in different languages entirely.

**BM25 over character 4-grams.** Telugu is agglutinative and sandhi fuses word
boundaries, so `యోగక్షేమం` and `యోగక్షేమాలను` share *no* whitespace token — but
ten 4-grams.

**Return source text, never the model's.** A citation is accepted when a long
contiguous run of the quote occurs verbatim in the verse; what is then displayed
is the **corpus text**, not the model's transcription. Drift is tolerated,
fabrication is not.

## Corpus

699 verses from [te.wikisource.org](https://te.wikisource.org) (CC BY-SA 4.0),
each with Sanskrit sloka and Telugu tatparyam. Three different source formats had
to be parsed; 16 of 18 chapters match canonical verse counts exactly, and the two
that don't are explained in [docs/CORPUS.md](docs/CORPUS.md).

Cleaning removed 15,247 characters of commentary leakage and Sanskrit bleed that
the retrieval eval surfaced — see `tests/test_corpus.py`, which guards against
both regressing.

## Install

```bash
pip install -e ".[dev]"          # library + tests
pip install -e ".[app]"          # + Gradio app
```

Requires Python 3.10+. Fine-tuned models are on the Hugging Face Hub; corpus and
precomputed vectors ship in `data/`.

## Use

```python
from telugu_gita_rag.retrieval.hybrid import search
hits = search("కోపం ఎలా పుడుతుంది?", top_k=8)
```

The deployed Space also exposes a plain HTTP API — no client library:

```
POST /gradio_api/call/answer   {"data": ["<question>"]}   → {"event_id"}
GET  /gradio_api/call/answer/{event_id}                   → SSE [answer, citations]
```

## Reproduce

```bash
python -m telugu_gita_rag.corpus.fetch          # scrape te.wikisource
python -m telugu_gita_rag.corpus.parse_gita     # → data/gita.jsonl
python -m telugu_gita_rag.corpus.clean          # strip bhashya + sloka bleed
python -m telugu_gita_rag.retrieval.build_index # embed + BM25
python -m telugu_gita_rag.evaluation.retrieval_eval
```

Fine-tuning runs on a free Colab T4 — see `notebooks/`.

## Docs

- [docs/CORPUS.md](docs/CORPUS.md) — sources, parsing, what is and isn't available
- [docs/FINETUNE.md](docs/FINETUNE.md) — contrastive training, hard negatives, why batch size is the quality lever
- [docs/FINDINGS.md](docs/FINDINGS.md) — measured results including the negative ones

## License

MIT for the code. Corpus text is CC BY-SA 4.0 from te.wikisource.org.
