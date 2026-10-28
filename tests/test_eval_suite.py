from types import SimpleNamespace

from app.eval_suite import _sample, norm, retrieval_metrics, summarize


def hit(source, text):
    return SimpleNamespace(source=source, text=text)


def test_norm_ignores_case_and_whitespace():
    assert norm("Up  to\n5 DAYS") == "up to 5 days"


def test_retrieval_metrics_partial_recall():
    case = {"evidence": [{"source": "a.md", "phrase": "24 days"},
                         {"source": "b.md", "phrase": "10 days"}]}
    hits = [hit("x.md", "nothing here"), hit("a.md", "You get 24  days of leave")]
    m = retrieval_metrics(case, hits)
    assert m["retrieval_recall"] == 0.5
    assert m["retrieval_hit"] == 1.0
    assert m["retrieval_mrr"] == 0.5          # first relevant chunk is at rank 2
    assert m["context_precision"] == 0.5


def test_phrase_from_the_wrong_document_does_not_count():
    case = {"evidence": [{"source": "a.md", "phrase": "24 days"}]}
    m = retrieval_metrics(case, [hit("other.md", "24 days")])
    assert m["retrieval_hit"] == 0.0 and m["retrieval_recall"] == 0.0


def test_no_evidence_means_no_retrieval_metrics():
    assert retrieval_metrics({"evidence": []}, [hit("a.md", "x")]) == {}


def test_summarize_groups_by_type_and_ignores_missing_values():
    rows = [
        {"id": "L1", "type": "lookup", "retrieval_recall": 1.0, "answer_correctness": 1.0},
        {"id": "L2", "type": "lookup", "retrieval_recall": 0.0, "answer_correctness": 0.5},
        {"id": "N1", "type": "no_answer", "answer_correctness": 1.0, "hallucinated": 0.0},
    ]
    s = summarize(rows)
    assert s["overall"]["n"] == 3
    assert s["lookup"]["retrieval_recall"] == 0.5
    assert s["lookup"]["answer_correctness"] == 0.75
    assert s["no_answer"]["retrieval_recall"] is None
    assert s["no_answer"]["hallucinated"] == 0.0


def test_sample_covers_every_type():
    cases = [{"id": f"{t}{i}", "type": t} for t in ("a", "b", "c") for i in range(5)]
    picked = _sample(cases, 6)
    assert len(picked) == 6
    assert {c["type"] for c in picked} == {"a", "b", "c"}
