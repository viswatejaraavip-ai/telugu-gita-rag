---
title: Telugu Bhagavad Gita RAG
emoji: 🪔
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
---

# భగవద్గీత — grounded Telugu Q&A

Answers only from the Bhagavad Gita's 699 verses. Every quotation is verified
verbatim against its source verse before display; unverified citations are dropped.

## Pipeline
fine-tuned BGE-M3 + char-4gram BM25 → RRF → fine-tuned cross-encoder rerank
→ route-aware grading → Haiku 4.5 generation → verbatim citation verification

## Secrets
- `ANTHROPIC_API_KEY` — generation
- `HF_TOKEN` — the model repos are private

## Variables
- `BIENCODER_REPO`, `RERANKER_REPO`
- `ENRICHMENT_VARIANT` — `full` | `filtered` | `off`
