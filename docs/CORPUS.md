# Telugu epics corpus — build report

Source: te.wikisource.org (MediaWiki API, CC BY-SA 4.0). Built 2026-09-11.

## What is usable today

| File | Records | Telugu tatparyam | Status |
|---|---|---|---|
| `corpus/gita.jsonl` | 705 verses | **Yes — 100%** | Ready for RAG |
| `corpus/ramayana.jsonl` | 13,693 slokas | **No — 0.5%** | Verse + citation layer only |

## Bhagavad Gita — ready

699 unique verses against a canonical 701, every one carrying a Telugu prose
tatparyam (mean 126 chars). **All 699 are citation-safe.**

Three source formats had to be handled — the Wikisource text is not uniform:

| Mode | Chapters | Shape | Verse numbers |
|---|---|---|---|
| A | 1, 2, 3, 13, 16 | full Sanskrit sloka + `\|\| c-v \|\|` + tatparyam | from the ref |
| B | 4-11, 14, 17, 18 | Sanskrit *incipit only* + tatparyam | positional |
| C | 12, 15 | incipit, then `N. <sanskrit> \|`, then tatparyam | from the `N.` prefix |

16 of 18 chapters match the canonical verse count exactly. The two that don't
are explained and harmless:

* **ch 1** repeated verse 13 (one copy had only commentary; deduplicated).
* **ch 16** is missing verse 4 in the source. The other 23 are numbered correctly.

`sloka_is_incipit_only: true` marks records whose Sanskrit is truncated to its
opening words (mode B). Join against sanskritdocuments.org on
(adhyaya, sloka) if you need the complete Sanskrit.

### Cleanup applied

`app/clean_corpus.py` fixed two defects that the retrieval eval surfaced — both
were polluting the embeddings:

* **Bhashya contamination.** 17 chapter-1 records carried Shankara / Ramanuja /
  Madhva commentary inside `tatparyam_te`, averaging 946 chars against a normal
  126. One record was *entirely* commentary with no gloss at all.
* **Sanskrit bleed-through.** 30 records across 7 chapters had the following
  verse's incipit or sloka appended to their tatparyam.

15,247 characters of junk removed; both defects now measure zero.

## Valmiki Ramayana — text yes, Telugu no

428 sargas present out of 637. Kandas 1-5 are complete; Yuddhakanda has 20/131
and Uttarakanda 2/100.

The blocker is not coverage, it is language: **only 64 of 13,693 slokas (0.5%)
have a Telugu tatparyam.** The rest is Sanskrit mechanically transliterated
into Telugu script — not Telugu. Balakanda sarga 1 has a gloss and is
unrepresentative of the rest.

Three sloka-reference formats were handled: `|1-40-1|`, Telugu digits
`|౧-౪౦-౧|`, and Sundarakanda's `|| 5.35.1`.

## Mahabharata — not built

1,806 chapters exist across 15/18 parvas, but they are the **Vyasa Mahabharata
in Sanskrit transliterated to Telugu script, with no Telugu translation** —
same problem as the Ramayana, and it is not Nannaya/Tikkana's Andhra
Mahabharatam. The Andhra Mahabharatam is not on Wikisource in usable form
(prefix scan returns 1 page).

Also available and not yet built: **Molla Ramayanam**, 96 pages already
segmented by episode with descriptive titles, in genuine 16th-c. literary
Telugu (no gloss).

## Record schema

Gita: `id, work, adhyaya_no, adhyaya, sloka_no, citation_en, citation_te,
sloka_sa, sloka_is_incipit_only, tatparyam_te, parse_mode,
verse_number_verified, source_page, source_url, license`

Ramayana: `id, work, kanda, kanda_no, sarga_no, sloka_no, citation_en,
citation_te, sloka_sa_in_telugu_script, tatparyam_te, has_tatparyam,
source_page, source_url, license`

## Rebuilding

    python3 scripts/gita.py        # writes corpus/gita.jsonl
    python3 scripts/ramayana.py    # writes corpus/ramayana.jsonl
    python3 scripts/audit.py       # tatparyam coverage report

Pages are cached under `cache/` — reruns do not re-fetch. The fetcher sends a
descriptive User-Agent and rate-limits to ~4 req/s, per Wikimedia etiquette.
