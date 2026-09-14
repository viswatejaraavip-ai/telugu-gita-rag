"""Regression tests for the invariants that actually broke in development."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pytest
from telugu_gita_rag.core import (FOREIGN, anchor_span, bm25_tokens, longest_verbatim_run,
                                  ndcg_at_k, parse_verse_id, rrf)

GOLD = "కర్మలు చేయడంలోనే నీకు అధికారం ఉన్నది. కర్మ ఫలాలపైన ఎప్పుడూ లేదు."

# --- citation id parsing: two production bugs lived here -------------------
@pytest.mark.parametrize("raw,want", [
    ("gita-2-47",      "gita-2-47"),
    ("[gita-2-47] ",   "gita-2-47"),   # bug 1: model kept the brackets
    (" 2-47 ",         "gita-2-47"),   # bug 2: model dropped the gita- prefix
    ("2.47",           "gita-2-47"),
    ("**gita-18-12**", "gita-18-12"),
    ("garbage",        None),
])
def test_parse_verse_id(raw, want):
    assert parse_verse_id(raw) == want

# --- anchoring: tolerate transcription drift, reject fabrication -----------
def test_anchor_accepts_one_char_drift():
    drifted = "కర్మలు చేయడంలోనే నీకు అధికారం ఉన్నది. కర్మ ఫలాలపై ఎప్పుడూ లేదు."
    span = anchor_span(drifted, GOLD)
    assert span is not None
    assert "ఫలాలపైన" in span, "must return SOURCE text, not the model's rendering"

def test_anchor_rejects_fabrication():
    assert anchor_span("పూర్తిగా అసంబంధిత వాక్యం ఇది ఎక్కడా లేదు", GOLD) is None

def test_anchor_rejects_short_incidental_overlap():
    assert anchor_span("కర్మ", GOLD) is None

# --- Telugu tokenisation ---------------------------------------------------
def test_char_grams_bridge_inflection():
    """యోగక్షేమం vs యోగక్షేమాలను share no whitespace token but many 4-grams."""
    a, b = "యోగక్షేమం ఎవరు వహిస్తారు?", "వారి యోగక్షేమాలను నేనే వహిస్తాను."
    assert not (set(a.split()) & set(b.split()))
    shared = {g for g in bm25_tokens(a) if len(g) == 4} & {g for g in bm25_tokens(b) if len(g) == 4}
    assert len(shared) >= 5

# --- fusion ----------------------------------------------------------------
def test_rrf_is_rank_based_and_dedupes():
    fused = rrf([["a", "b"], ["b", "a"]])
    assert fused["a"] == pytest.approx(fused["b"], rel=1e-9)

def test_rrf_rewards_appearing_in_both_lists():
    fused = rrf([["a", "x"], ["a", "y"]])
    assert fused["a"] > fused["x"]

# --- script contamination guard -------------------------------------------
def test_foreign_script_detected():
    assert FOREIGN.search("మనुష్యుడు")          # Devanagari vowel sign
    assert FOREIGN.search("ప్రశాంáfrica")        # Latin fragment
    assert not FOREIGN.search("కర్మ ఫలాలపైన ఎప్పుడూ లేదు")

# --- metrics ---------------------------------------------------------------
def test_ndcg_rank_one_is_perfect():
    assert ndcg_at_k(["g"], {"g"}) == pytest.approx(1.0)

def test_ndcg_zero_when_absent():
    assert ndcg_at_k(["a", "b"], {"g"}) == 0.0
