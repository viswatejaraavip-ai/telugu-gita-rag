# Fine-tuning BGE-M3 for Telugu Gita retrieval

## Why

The floor test (verbatim substring → must rank #1) scored **26/30 rank-1, 0
misses in top 20**, and a topic-matched anisotropy check put Telugu at 0.485
against English 0.502. The encoder is not weak on Telugu.

What fails is **question-to-statement asymmetry**: queries are interrogative,
documents are 126-character declaratives where the answer is one clause among
several. Contrastive fine-tuning targets exactly that — it reshapes the space so
"does this verse answer this question?" becomes the similarity being measured.

## Pipeline

    app/split_corpus.py    → train_verses.jsonl / held_verses.jsonl
    app/gen_queries.py     → train_queries.jsonl / held_queries.jsonl   (Claude)
    app/mine_negatives.py  → train_triplets.jsonl
    finetune/finetune_bge_m3_gita.ipynb  → Colab, T4

### 1. Split — by VERSE, not by query

699 verses → 581 train / 104 held-out, plus 14 verses that are gold answers for
the hand-written eval questions, excluded from **both**.

Holding out verses is what makes the eval honest. Held-out *queries* drawn from
training *verses* measure memorisation, and will show a large fake gain.

### 2. Generate queries

Five per verse, via `claude-opus-5`:

* 2 `lookup` — direct factual questions
* 2 `thematic` — first-person life situations that never mention the Gita
* 1 `tenglish` — romanized Telugu, as people actually type on phones

The prompt forbids reusing the verse's own vocabulary. A query that echoes the
commentary's wording trains the model on lexical overlap, which BM25 already
handles — the dense side needs to learn the *paraphrase*.

Including `tenglish` in training is what gives any chance of fixing the 0.00
romanized recall without a transliteration hop.

Resumable: already-generated verse ids are skipped, so an interrupted run costs
nothing.

### 3. Mine hard negatives

Random negatives teach almost nothing — a verse about Arjuna's despair is
trivially separable from a karma query. Hard negatives are verses the *current*
retriever ranks highly for a query but which are wrong. Those are the confusions
actually costing recall.

Negatives are drawn only from the training split, so held-out text never leaks
into training.

### 4. Train

`CachedMultipleNegativesRankingLoss` (GradCache) decouples effective batch size
from GPU memory, so a free T4 trains at batch 64 instead of 8. **Batch size is
the main quality lever** — it sets how many wrong answers each query is
contrasted against. `mini_batch_size` is the memory knob and does not change the
effective batch; lower it on OOM.

`learning_rate=2e-5` and 3 epochs are deliberately conservative. A few thousand
examples will happily overfit into a model that knows this corpus and has
forgotten general Telugu. The evaluator runs every 25 steps and
`load_best_model_at_end` takes the best checkpoint rather than the last.

## Measuring

Baseline on the 14 hand-written questions, before any of this:

    recall@1=0.29  recall@5=0.43  recall@10=0.57  MRR=0.367
      lookup    n=9  recall@10=0.78  median_rank=2
      thematic  n=3  recall@10=0.33  median_rank=16
      tenglish  n=2  recall@10=0.00

Expect movement on **lookup** and, if the tenglish examples do their job, on
**tenglish**. **Thematic** is an abstraction problem that the enrichment layer
addresses more directly — do not expect fine-tuning alone to fix it.

After training, re-run `app/evaluate.py`. Those 14 questions are hand-written and
their gold verses never entered training, so they are the honest test set.
Also confirm general Telugu retrieval has not degraded.

## Applying the result

Unzip the downloaded model, point `MODEL_NAME` in `app/retrieval.py` at the
folder, then:

    HF_HUB_DISABLE_XET=1 ./.venv/bin/python app/build_index.py
    ./.venv/bin/python app/evaluate.py
