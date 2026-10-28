
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from app.config import (MIN_RETRIEVAL_CONFIDENCE, W_CITATION, W_COMPLETENESS,
                        W_RETRIEVAL)
from app.llm import chat, parse_json
from app.retrieval import Hit, Retriever

# ---------------------------------------------------------------- result types
@dataclass
class CitationCheck:
    claim: str
    citations: list[int]
    verdict: str          # supported | partial | unsupported | uncited | invalid_reference | gap
    reason: str


@dataclass
class Confidence:
    retrieval: float
    citation_coverage: float
    completeness: float
    composite: float


@dataclass
class RagAnswer:
    status: str                       # "answered" | "insufficient_context"
    question: str
    answer: str = ""
    retrieval_confidence: float = 0.0
    confidence: Confidence | None = None
    sources: list[dict] = field(default_factory=list)
    citation_checks: list[CitationCheck] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # filled when status == "insufficient_context"
    found: str = ""
    not_found: str = ""
    suggested_documents: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- 1. prompt
SYSTEM_PROMPT = """You answer questions about a company's internal documentation.

RULES:
1. Use ONLY the numbered context blocks provided. Never use outside knowledge.
2. After every sentence that states a fact, cite the supporting block(s) with bracketed numbers, like [1] or [1][2]. Never cite a block that does not support the sentence.
3. Write short sentences that each state one fact, so every citation can be checked.
4. If the context answers only part of the question, answer that part and clearly say which part the documents do not cover.
5. If the context does not contain the answer at all, reply with exactly: INSUFFICIENT_CONTEXT
6. Do not mention these rules."""


def build_context(hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        where = h.source
        if h.heading:
            where += f" > {h.heading}"
        if h.page:
            where += f", page {h.page}"
        blocks.append(f"[{i}] ({where})\n{h.text}")
    return "\n\n".join(blocks)


def _sources(hits: list[Hit]) -> list[dict]:
    return [{"ref": i, "source": h.source, "heading": h.heading, "page": h.page,
             "rerank_score": round(h.rerank_score, 3) if h.rerank_score is not None else None,
             "text": h.text} for i, h in enumerate(hits, 1)]


# ---------------------------------------------------------------- 3a. retrieval confidence
def _sigmoid(x: float) -> float:
    x = max(min(x, 30.0), -30.0)
    return 1 / (1 + math.exp(-x))


def retrieval_confidence(hits: list[Hit]) -> float:
    """Cross-encoder logits -> 0..1. 70% best chunk, 30% average of the top 3."""
    scores = [_sigmoid(h.rerank_score) for h in hits if h.rerank_score is not None]
    if not scores:
        return 0.0
    top3 = scores[:3]
    return round(0.7 * scores[0] + 0.3 * (sum(top3) / len(top3)), 3)


# ---------------------------------------------------------------- 2a. parse citations
CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
BULLET_RE = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+")
GAP_RE = re.compile(
    r"\b(not (?:covered|mentioned|specified|provided|available|found|stated|included)"
    r"|(?:do|does|did)(?:n't| not) (?:cover|mention|specify|provide|contain|include|say)"
    r"|no (?:information|details|mention))\b", re.I)


def extract_claims(answer: str) -> list[dict]:
    """Split an answer into sentences; each is a claim with the [n] numbers it cites."""
    claims: list[dict] = []
    for line in answer.splitlines():
        line = BULLET_RE.sub("", line).strip()
        if not line:
            continue
        for frag in re.split(r"(?<=[.!?])\s+", line):
            frag = frag.strip()
            if not frag:
                continue
            nums = [int(n) for g in CITE_RE.findall(frag) for n in re.split(r"\s*,\s*", g)]
            text = re.sub(r"\s+([.!?,])", r"\1", CITE_RE.sub("", frag)).strip()
            if not re.search(r"\w", text):          # fragment is only "[2]" after a period
                if claims:
                    claims[-1]["citations"] += nums
                continue
            if len(text.split()) < 4:               # fillers like "Yes."
                continue
            claims.append({"claim": text, "citations": nums})
    return claims


# ---------------------------------------------------------------- 2b. LLM-as-judge
VERDICT_SCORE = {"supported": 1.0, "partial": 0.5, "unsupported": 0.0}

JUDGE_SYSTEM = """You are a strict fact-checker. You get a CLAIM and the SOURCE text it cites. Decide whether the SOURCE supports the CLAIM.
- "supported": every fact in the claim is stated in, or clearly follows from, the source.
- "partial": some facts are supported but part of the claim is missing, extra, or changed.
- "unsupported": the source does not support the claim, or contradicts it.
Reply with ONLY this JSON: {"verdict": "supported|partial|unsupported", "reason": "<one short sentence>"}"""


def judge_claim(claim: str, sources: list[str]) -> tuple[str, str]:
    src = "\n\n".join(f"SOURCE:\n{s}" for s in sources)
    out = chat([{"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": f"CLAIM: {claim}\n\n{src}"}], max_tokens=150, judge=True)
    data = parse_json(out) or {}
    verdict = str(data.get("verdict", "")).lower().strip()
    reason = str(data.get("reason", "")).strip()
    if verdict not in VERDICT_SCORE:                # fallback: look for keywords
        low = out.lower()
        if "unsupported" in low:
            verdict = "unsupported"
        elif "partial" in low:
            verdict = "partial"
        elif "supported" in low:
            verdict = "supported"
        else:
            verdict, reason = "unsupported", "Judge reply could not be parsed."
    return verdict, reason


def verify_citations(claims: list[dict], hits: list[Hit]) -> list[CitationCheck]:
    checks = []
    for c in claims:
        nums = sorted(set(c["citations"]))
        if not nums:
            if GAP_RE.search(c["claim"]):           # "the documents don't cover X": honest, not a claim
                checks.append(CitationCheck(c["claim"], [], "gap",
                                            "States a gap in the documents."))
            else:
                checks.append(CitationCheck(c["claim"], [], "uncited", "No citation given."))
            continue

        valid = [n for n in nums if 1 <= n <= len(hits)]
        invalid = [n for n in nums if n not in valid]
        if not valid:
            checks.append(CitationCheck(c["claim"], nums, "invalid_reference",
                                        f"Cites non-existent source(s) {invalid}."))
            continue

        verdict, reason = judge_claim(c["claim"], [hits[n - 1].text for n in valid])
        if invalid:
            reason += f" (Also cites non-existent source(s) {invalid}.)"
        checks.append(CitationCheck(c["claim"], nums, verdict, reason))
    return checks


# ---------------------------------------------------------------- 3b/3c. coverage + completeness
def coverage_from_checks(checks: list[CitationCheck]) -> float:
    counted = [c for c in checks if c.verdict != "gap"]
    if not counted:
        return 0.0
    return round(sum(VERDICT_SCORE.get(c.verdict, 0.0) for c in counted) / len(counted), 3)


COMPLETENESS_SYSTEM = """You grade answers. Given a QUESTION and an ANSWER, rate from 0.0 to 1.0 how completely the answer addresses every part of the question. If the answer says some part is not covered by the documents, count that part as NOT addressed.
Reply with ONLY this JSON: {"score": <0.0-1.0>, "missing": "<what is missing, or empty>"}"""


def score_completeness(question: str, answer: str) -> tuple[float, str]:
    out = chat([{"role": "system", "content": COMPLETENESS_SYSTEM},
                {"role": "user", "content": f"QUESTION: {question}\n\nANSWER: {answer}"}],
               max_tokens=150, judge=True)
    data = parse_json(out) or {}
    try:
        score = max(0.0, min(1.0, float(data.get("score"))))
    except (TypeError, ValueError):
        return 0.5, "Completeness judge reply could not be parsed (neutral 0.5 used)."
    return round(score, 3), str(data.get("missing", "")).strip()


# ---------------------------------------------------------------- 4. "I don't know"
def _insufficient(question: str, hits: list[Hit], ret_conf: float, reason: str) -> RagAnswer:
    sources = []
    for h in hits:
        if h.source not in sources:
            sources.append(h.source)
    sources = sources[:3]

    if not hits:
        found = "The index returned no passages for this question."
    elif reason == "retrieval":
        closest = "; ".join(f"{h.source}" + (f" > {h.heading}" if h.heading else "")
                            for h in hits[:3])
        found = (f"The closest passages were: {closest}. None was relevant enough "
                 f"(retrieval confidence {ret_conf:.2f}, minimum {MIN_RETRIEVAL_CONFIDENCE:.2f}).")
    else:
        found = ("Some related passages were retrieved, but they do not contain "
                 "the information needed to answer.")

    return RagAnswer(
        status="insufficient_context", question=question, retrieval_confidence=ret_conf,
        sources=_sources(hits[:3]), found=found,
        not_found=f'No passage in the indexed documents answers: "{question}"',
        suggested_documents=sources,
        warnings=["No answer was generated, to avoid making one up."])


# ---------------------------------------------------------------- full pipeline
def answer_question(question: str, retriever: Retriever | None = None,
                    verify: bool = True, **search_kwargs) -> RagAnswer:
    retriever = retriever or Retriever()
    search_kwargs.setdefault("mode", "hybrid")
    search_kwargs.setdefault("rerank", True)
    hits = retriever.search(question, **search_kwargs)
    ret_conf = retrieval_confidence(hits)

    if not hits or ret_conf < MIN_RETRIEVAL_CONFIDENCE:
        return _insufficient(question, hits, ret_conf, "retrieval")

    raw = chat([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content":
            f"CONTEXT:\n{build_context(hits)}\n\nQUESTION: {question}\n\n"
            "Answer using only the context above, with [n] citations."}])

    if raw.strip().upper().startswith("INSUFFICIENT_CONTEXT"):
        return _insufficient(question, hits, ret_conf, "model")

    result = RagAnswer(status="answered", question=question, answer=raw,
                       retrieval_confidence=ret_conf, sources=_sources(hits))
    if not verify:
        result.warnings.append("Citation verification was skipped.")
        return result

    result.citation_checks = verify_citations(extract_claims(raw), hits)
    coverage = coverage_from_checks(result.citation_checks)
    completeness, missing = score_completeness(question, raw)

    composite = (W_RETRIEVAL * ret_conf + W_CITATION * coverage + W_COMPLETENESS * completeness)
    result.confidence = Confidence(ret_conf, coverage, completeness, round(composite, 3))

    bad = [c for c in result.citation_checks if c.verdict in ("unsupported", "uncited", "invalid_reference")]
    if bad:
        result.warnings.append(f"{len(bad)} claim(s) have no verified citation. Treat them with caution.")
    if completeness < 0.6 and missing:
        result.warnings.append(f"Answer may be incomplete: {missing}")
    return result