"""The corpus is the product; these guard its integrity."""
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pytest
from telugu_gita_rag.core import FOREIGN, TELUGU

DATA = Path(__file__).resolve().parents[1] / "data" / "gita.jsonl"

@pytest.fixture(scope="module")
def corpus():
    return [json.loads(l) for l in DATA.open(encoding="utf-8") if l.strip()]

def test_size(corpus):
    assert 690 <= len(corpus) <= 701

def test_required_fields(corpus):
    need = {"id", "adhyaya_no", "sloka_no", "tatparyam_te", "citation_te",
            "source_url", "verse_number_verified"}
    for r in corpus:
        assert need <= set(r), f"{r.get('id')} missing {need - set(r)}"

def test_ids_unique(corpus):
    ids = [r["id"] for r in corpus]
    assert len(ids) == len(set(ids))

def test_every_verse_has_telugu_gloss(corpus):
    for r in corpus:
        assert len(TELUGU.findall(r["tatparyam_te"])) >= 8, r["id"]

def test_no_commentary_leakage(corpus):
    """Chapter 1 once carried Shankara/Ramanuja bhashya inside tatparyam_te."""
    bad = re.compile(r"ఆదిశంకరాచార్యుల|రామానుజాచార్యుల|మధ్వాచార్యుల")
    assert [r["id"] for r in corpus if bad.search(r["tatparyam_te"])] == []

def test_no_sanskrit_bleed(corpus):
    """The next verse's sloka used to bleed into the previous tatparyam."""
    assert [r["id"] for r in corpus if "|" in r["tatparyam_te"]] == []

def test_no_foreign_script(corpus):
    assert [r["id"] for r in corpus if FOREIGN.search(r["tatparyam_te"])] == []

def test_adhyaya_range(corpus):
    assert all(1 <= r["adhyaya_no"] <= 18 for r in corpus)
