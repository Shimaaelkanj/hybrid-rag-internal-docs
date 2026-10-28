from dashboard.helpers import band, confidence_rows, link_citations, overlap, source_label

SOURCES = [{"ref": 1, "source": "hr.md", "heading": "Annual Leave", "page": None},
           {"ref": 2, "source": "x.pdf", "heading": "", "page": 4}]


def test_citations_become_links_to_source_cards():
    html = link_citations("You get 24 days [1]. See also [2].", SOURCES, prefix="hybrid")
    assert 'href="#hybrid-1"' in html and 'href="#hybrid-2"' in html
    assert "hr.md &gt; Annual Leave" in html and "x.pdf (p.4)" in html   # tooltip is HTML-escaped


def test_multiple_citations_and_missing_sources():
    html = link_citations("Fact [1, 2] and wrong [9].", SOURCES)
    assert html.count('class="cite"') == 2
    assert 'class="cite-bad"' in html


def test_answer_text_is_html_escaped():
    html = link_citations("<script>alert(1)</script> [1]", SOURCES)
    assert "<script>" not in html and "&lt;script&gt;" in html


def test_source_label_variants():
    assert source_label(SOURCES[0]) == "hr.md > Annual Leave"
    assert source_label({"source": "a.md"}) == "a.md"


def test_confidence_rows_and_bands():
    rows = confidence_rows({"retrieval": 0.8, "citation_coverage": 1.0, "completeness": 0.5})
    assert [r[1] for r in rows] == [0.8, 1.0, 0.5]
    assert band(0.9) == "High" and band(0.6) == "Medium" and band(0.2) == "Low"


def test_overlap():
    d = overlap(["a", "b", "c"], ["b", "d"])
    assert d == {"only_first": ["a", "c"], "only_second": ["d"], "both": ["b"]}
