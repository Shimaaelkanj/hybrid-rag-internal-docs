
from app.generation import (CitationCheck, coverage_from_checks, extract_claims,
                            retrieval_confidence)
from app.retrieval import Hit


def hit(score):
    return Hit(id="x", text="t", source="s", heading="", page=None, strategy="recursive",
               rerank_score=score)


def test_citation_before_and_after_period():
    claims = extract_claims("The default pool size is 10 [1]. The maximum is 100. [2]")
    assert claims[0]["citations"] == [1]
    assert claims[1]["citations"] == [2]
    assert claims[0]["claim"] == "The default pool size is 10."


def test_bullets_and_multiple_citations():
    claims = extract_claims("- Passwords must be 14 characters long [1, 2].\n- Rotate them every 90 days [2][3].")
    assert claims[0]["citations"] == [1, 2]
    assert claims[1]["citations"] == [2, 3]


def test_coverage_ignores_gap_statements():
    checks = [CitationCheck("a b c d", [1], "supported", ""),
              CitationCheck("e f g h", [], "uncited", ""),
              CitationCheck("the documents do not cover x", [], "gap", "")]
    assert coverage_from_checks(checks) == 0.5


def test_retrieval_confidence_range():
    assert retrieval_confidence([hit(8.0)]) > 0.9
    assert retrieval_confidence([hit(-8.0)]) < 0.05
    assert retrieval_confidence([]) == 0.0