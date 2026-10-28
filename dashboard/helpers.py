"""Pure helper functions for the dashboard (no Streamlit import, so they are easy to test)."""
from __future__ import annotations

import html
import re

CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def source_label(src: dict) -> str:
    label = src["source"]
    if src.get("heading"):
        label += f" > {src['heading']}"
    if src.get("page"):
        label += f" (p.{src['page']})"
    return label


def link_citations(answer: str, sources: list[dict], prefix: str = "src") -> str:
    """HTML for the answer in which every [n] links to the source card with id '<prefix>-n'.

    The text is HTML-escaped first, so nothing the model writes can inject markup.
    Citations that point to a non-existent source are marked instead of linked.
    """
    by_ref = {s["ref"]: s for s in sources}

    def replace(match: re.Match) -> str:
        parts = []
        for n in (int(x) for x in re.split(r"\s*,\s*", match.group(1))):
            if n in by_ref:
                tip = html.escape(source_label(by_ref[n]), quote=True)
                parts.append(f'<a class="cite" href="#{prefix}-{n}" title="{tip}">[{n}]</a>')
            else:
                parts.append(f'<span class="cite-bad" title="This source does not exist">[{n}]</span>')
        return "".join(parts)

    return CITE_RE.sub(replace, html.escape(answer)).replace("\n", "<br>")


def confidence_rows(conf: dict) -> list[tuple[str, float, str]]:
    """(label, value 0-1, explanation) for each confidence dimension."""
    return [
        ("Retrieval confidence", conf["retrieval"], "How relevant the retrieved chunks were to the question."),
        ("Citation coverage", conf["citation_coverage"], "Share of claims backed by a verified citation."),
        ("Completeness", conf["completeness"], "Whether every part of the question was addressed."),
    ]


def band(value: float) -> str:
    return "High" if value >= 0.75 else "Medium" if value >= 0.5 else "Low"


def overlap(first: list[str], second: list[str]) -> dict[str, list[str]]:
    """Which chunk ids are only in the first list, only in the second, or in both."""
    a, b = set(first), set(second)
    return {"only_first": [i for i in first if i not in b],
            "only_second": [i for i in second if i not in a],
            "both": [i for i in first if i in b]}


VERDICT_ICON = {"supported": "✅", "partial": "🟡", "unsupported": "❌", "uncited": "⚠️",
                "invalid_reference": "❌", "gap": "ℹ️"}
