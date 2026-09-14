# Findings

Measured results from building this, including the ones that didn't work.
Negative results are kept because each one cost real time to establish.

## Retrieval

**Fine-tuning the bi-encoder: +46% recall@10** (0.350 → 0.512 on 320 held-out
queries). Trained on 5,810 triplets from 2,905 Claude-generated Telugu queries.
The contrastive margin on held triplets went from +0.000 to +0.345 — and almost
all of that came from pushing negatives *down* (0.500 → 0.177) rather than
pulling positives up (0.500 → 0.522).

**Fine-tuning the cross-encoder: +24% on top** (0.512 → 0.634). The stock
reranker *hurt* — it improved only `lookup`, the query type closest to generic
retrieval, and degraded the two types fine-tuning had taught. A reranker only
helps if it is better at the task than the retriever it reorders.

**Reranking pool saturates at 30.** Measured by reranking the top 50 once and
truncating:

| pool | recall@10 | NDCG@10 | ceiling | captured |
|---|---|---|---|---|
| 15 | 0.562 | 0.436 | 0.588 | 96% |
| 30 | 0.625 | 0.462 | 0.762 | 82% |
| 50 | 0.625 | 0.467 | 0.812 | 77% |

A bigger pool raises the ceiling faster than the reranker can climb it —
candidates at RRF ranks 30–50 are ones the retriever already scored poorly.

## Things that did not work

**Query expansion made retrieval worse.** The router wrote up to 3 Telugu search
phrases and results were merged by score: recall@1 **0.300 → 0.160**, NDCG 0.369
→ 0.288, at 3.4× the latency. The cause was the merge — scores from separate
reranker runs on different queries are not comparable. Fusing by *rank* (RRF)
might work; merging by score destroyed the ordering while leaving recall@10 flat,
which is the signature of a ranking bug rather than a retrieval one.

**The thematic enrichment layer is unproven.** Generating themes and first-person
life-situations for every verse lifted thematic recall@50 (0.727 → 0.773) but not
recall@10, and with a reranker in the pipeline the effect vanished entirely —
`B == A` to three decimals across 60 queries.

**Selective enrichment actively hurt.** Filtering to the ~50% of verses with a
"life application" removed **55% of held-out thematic gold verses**. The skip
rules were written for practical problems (work, anger) and excluded
"enumeration of divine attributes" and "metaphysical description" — but a large
share of thematic queries are existential (*who is God, what am I*), and those
verses are their answers. Even chapter 1 is over-represented as thematic gold,
because Arjuna's breakdown is the most relatable passage in the book.

**Prompt engineering did not reduce unsupported claims.** Four explicit
prohibitions with concrete Telugu examples, targeting four observed drift modes:
76% → 76%. Switching Haiku → Sonnet did what prompting could not (76% → 33%).
The drift was capability-limited.

## Generation

**Latency 45.1s → ~10s**, first word at 3.1s server-side:

| change | effect |
|---|---|
| thinking budget 2000 → 0 | generation 34.2s → 10.5s |
| folded `grade` into `generate` | −4 to 6s (one less round trip) |
| parallel route ‖ retrieve | ready 6.0s → 2.4s |
| streaming | first word at 3.1s instead of after the full answer |

**Small models corrupt low-resource scripts.** Haiku emitted Devanagari vowel
signs and once a Latin fragment mid-word in Telugu (`మనुష్యుడు`,
`ప్రశాంáfrica`) in ~3% of answers. Telugu and Devanagari share Brahmi-descended
vowel signs and Telugu tokenizes into many pieces. Sonnet: zero.

## Method notes

**Measure the system you ship.** Every retrieval number was initially measured on
`search(question)` while the graph actually searched LLM-rewritten queries — a
different system. That gap hid a 47% recall@1 regression that no offline metric
would have caught.

**Silent failures need counters, not just error logs.** Citation verification
dropped to 0% after a refactor and still rendered perfectly formed answers. Two
parser bugs (bracketed ids, then a missing `gita-` prefix) discarded valid
citations. Both were found by a `cited N` counter and an `unmatched id` log added
while chasing the wrong hypothesis.

**Small evals mislead.** A pool comparison on 50 queries produced sample noise
larger than the effect being measured, leading to a conclusion that a
fixed-query-set rerun overturned.
