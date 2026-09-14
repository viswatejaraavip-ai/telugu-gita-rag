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
