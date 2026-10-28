"""Checks that the golden dataset is well-formed and consistent with the corpus.

Pure Python: no models, no LLM. Run it every time you edit eval/golden.json or the corpus.
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = json.loads((ROOT / "eval" / "golden.json").read_text(encoding="utf-8"))
TYPES = {"lookup", "multi_hop", "no_answer", "ambiguous"}


def norm(text):
    return " ".join(text.lower().split())


def corpus_bodies():
    """Corpus text WITHOUT heading lines, because chunks contain body text only."""
    bodies = {}
    for path in (ROOT / "eval" / "corpus").glob("*.md"):
        lines = path.read_text(encoding="utf-8").splitlines()
        bodies[path.name] = norm("\n".join(l for l in lines if not l.startswith("#")))
    return bodies


def test_at_least_50_cases_with_all_types():
    assert len(GOLDEN) >= 50
    assert {c["type"] for c in GOLDEN} == TYPES


def test_ids_are_unique_and_fields_present():
    assert len({c["id"] for c in GOLDEN}) == len(GOLDEN)
    for c in GOLDEN:
        assert c["question"].strip() and c["golden_answer"].strip()
        assert c["type"] in TYPES


def test_every_type_has_enough_cases():
    counts = Counter(c["type"] for c in GOLDEN)
    assert counts["lookup"] >= 20 and counts["multi_hop"] >= 8
    assert counts["no_answer"] >= 8 and counts["ambiguous"] >= 6


def test_evidence_phrases_exist_in_the_corpus():
    bodies = corpus_bodies()
    for c in GOLDEN:
        for e in c["evidence"]:
            assert e["source"] in bodies, f"{c['id']}: unknown source {e['source']}"
            assert norm(e["phrase"]) in bodies[e["source"]], f"{c['id']}: phrase not found: {e['phrase']}"


def test_no_answer_cases_have_no_evidence_and_others_do():
    for c in GOLDEN:
        if c["type"] == "no_answer":
            assert c["evidence"] == []
        else:
            assert c["evidence"], f"{c['id']} needs evidence"


def test_multi_hop_cases_need_at_least_two_evidence_items():
    for c in GOLDEN:
        if c["type"] == "multi_hop":
            assert len(c["evidence"]) >= 2, c["id"]
