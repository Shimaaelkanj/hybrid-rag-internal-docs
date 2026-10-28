"""Evaluation suite.

For every golden test case it measures:
  * retrieval relevance : were the chunks that contain the evidence retrieved? (no LLM)
  * answer correctness  : LLM-as-judge compares the answer with the golden answer
  * faithfulness        : is every claim in the answer supported by the retrieved context?
  * citation accuracy   : does the cited chunk really support the claim it is attached to?
and for questions with no answer in the corpus: did the system correctly refuse?
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

from app.config import (CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR, DUPLICATES_FILE,
                        EMBEDDING_PROVIDER, JUDGE_MODEL, LLM_MODEL,
                        MIN_RETRIEVAL_CONFIDENCE, ROOT)
from app.generation import (GAP_RE, VERDICT_SCORE, answer_question, extract_claims,
                            judge_claim, retrieval_confidence, verify_citations)
from app.indexing import get_collection, load_chunk_rows
from app.ingest import ingest_path
from app.llm import chat, parse_json
from app.retrieval import Retriever

EVAL_DIR = ROOT / "eval"
CORPUS_DIR = EVAL_DIR / "corpus"
GOLDEN_FILE = EVAL_DIR / "golden.json"
RESULTS_DIR = EVAL_DIR / "results"

HEADLINE = ["retrieval_recall", "answer_correctness", "faithfulness", "citation_accuracy"]
METRICS = ["retrieval_recall", "retrieval_hit", "retrieval_mrr", "context_precision",
           "answer_correctness", "faithfulness", "citation_accuracy", "citation_coverage",
           "false_refusal", "hallucinated", "latency_s"]


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def norm(text: str) -> str:
    """Lower-case and collapse whitespace so evidence matching ignores formatting."""
    return " ".join(text.lower().split())


# ------------------------------------------------------------------ retrieval relevance
def _chunk_has(hit, evidence: dict) -> bool:
    return hit.source == evidence["source"] and norm(evidence["phrase"]) in norm(hit.text)


def retrieval_metrics(case: dict, hits: list) -> dict:
    """A chunk is relevant if it comes from the evidence document AND contains the evidence phrase.

    Matching on the phrase (not on a chunk id) keeps the metric comparable across chunking
    strategies, where chunk boundaries differ.
    """
    evidence = case["evidence"]
    if not evidence:
        return {}
    first_ranks = []
    for e in evidence:
        rank = next((i for i, h in enumerate(hits, 1) if _chunk_has(h, e)), None)
        if rank:
            first_ranks.append(rank)
    relevant_chunks = sum(1 for h in hits if any(_chunk_has(h, e) for e in evidence))
    return {
        "retrieval_recall": len(first_ranks) / len(evidence),       # share of evidence found
        "retrieval_hit": 1.0 if first_ranks else 0.0,                # any evidence found
        "retrieval_mrr": 1.0 / min(first_ranks) if first_ranks else 0.0,
        "context_precision": relevant_chunks / len(hits) if hits else 0.0,
    }


# ------------------------------------------------------------------ answer correctness (judge)
CORRECTNESS_SYSTEM = """You are grading an answer to a question about company documents.
You get the QUESTION, the GOLDEN ANSWER (the reference) and the CANDIDATE ANSWER.
Decide whether the candidate agrees with the golden answer on every key fact (numbers, names, conditions). Extra correct detail is fine. Ignore citation markers such as [1].
- "correct": all key facts of the golden answer are present and right.
- "partial": some key facts are right but others are missing or wrong.
- "incorrect": key facts are wrong or contradicted, or the answer does not address the question.
Reply with ONLY this JSON: {"verdict": "correct|partial|incorrect", "reason": "<one short sentence>"}"""

CORRECTNESS_SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


def judge_correctness(question: str, golden: str, candidate: str) -> tuple[float, str]:
    out = chat([{"role": "system", "content": CORRECTNESS_SYSTEM},
                {"role": "user", "content":
                    f"QUESTION: {question}\n\nGOLDEN ANSWER: {golden}\n\nCANDIDATE ANSWER: {candidate}"}],
               max_tokens=150, judge=True)
    data = parse_json(out) or {}
    verdict = str(data.get("verdict", "")).lower().strip()
    reason = str(data.get("reason", "")).strip()
    if verdict not in CORRECTNESS_SCORE:
        low = out.lower()                      # check "incorrect" before "correct"
        if "incorrect" in low:
            verdict = "incorrect"
        elif "partial" in low:
            verdict = "partial"
        elif "correct" in low:
            verdict = "correct"
        else:
            verdict, reason = "incorrect", "Judge reply could not be parsed."
    return CORRECTNESS_SCORE[verdict], reason


# ------------------------------------------------------------------ one test case
def run_case(case: dict, retriever, retrieval_only: bool = False,
             faithfulness: bool = True) -> dict:
    start = time.perf_counter()
    question = case["question"]
    answerable = bool(case["evidence"])

    hits = retriever.search(question, mode="hybrid", rerank=True)
    conf = retrieval_confidence(hits)
    row = {"id": case["id"], "type": case["type"], "question": question,
           "retrieval_confidence": conf, **retrieval_metrics(case, hits)}

    # Fast, LLM-free mode: only the retrieval metrics, plus what the confidence gate would do.
    if retrieval_only:
        gate_refused = (not hits) or conf < MIN_RETRIEVAL_CONFIDENCE
        if answerable:
            row["false_refusal"] = float(gate_refused)
        else:
            row["hallucinated"] = float(not gate_refused)     # upper bound: the LLM may still abstain
        row["latency_s"] = round(time.perf_counter() - start, 2)
        return row

    result = answer_question(question, retriever, verify=False)
    row["status"], row["answer"] = result.status, result.answer

    refused = result.status == "insufficient_context"
    abstained = refused or bool(GAP_RE.search(result.answer or ""))

    # ---- answer correctness
    if not answerable:                                        # correct behaviour = abstain
        row["hallucinated"] = float(not abstained)
        row["answer_correctness"] = 1.0 - row["hallucinated"]
    elif refused:                                             # refused a question it could answer
        row["false_refusal"] = 1.0
        row["answer_correctness"] = 0.0
        row["correctness_reason"] = "The system refused to answer an answerable question."
    else:
        row["false_refusal"] = 0.0
        row["answer_correctness"], row["correctness_reason"] = judge_correctness(
            question, case["golden_answer"], result.answer)

    # ---- faithfulness and citation accuracy
    if result.status == "answered":
        sources = [SimpleNamespace(text=s["text"]) for s in result.sources]
        checks = verify_citations(extract_claims(result.answer), sources)
        factual = [c for c in checks if c.verdict != "gap"]    # "docs do not cover X" is not a claim
        cited = [c for c in factual if c.citations]
        row["n_claims"] = len(factual)
        row["citation_accuracy"] = mean(VERDICT_SCORE.get(c.verdict, 0.0) for c in cited)
        row["citation_coverage"] = len(cited) / len(factual) if factual else None

        if faithfulness and factual:
            context = [s.text for s in sources]                # ALL retrieved chunks, ignoring citations
            row["faithfulness"] = mean(VERDICT_SCORE[judge_claim(c.claim, context)[0]]
                                       for c in factual)
        row["claims"] = [{"claim": c.claim, "citations": c.citations,
                          "verdict": c.verdict, "reason": c.reason} for c in checks]

    row["latency_s"] = round(time.perf_counter() - start, 2)
    return row


# ------------------------------------------------------------------ aggregation
def _aggregate(rows: list[dict]) -> dict:
    out = {"n": len(rows)}
    for m in METRICS:
        v = mean(r.get(m) for r in rows)
        out[m] = None if v is None else round(v, 4)
    parts = [out[m] for m in HEADLINE if out.get(m) is not None]
    out["overall_score"] = round(sum(parts) / len(parts), 4) if parts else None
    return out


def summarize(rows: list[dict]) -> dict:
    summary = {"overall": _aggregate(rows)}
    for t in sorted({r["type"] for r in rows}):
        summary[t] = _aggregate([r for r in rows if r["type"] == t])
    return summary


COLS = [("n", "n", 4), ("retrieval_recall", "recall", 8), ("context_precision", "ctx_prec", 10),
        ("answer_correctness", "correct", 9), ("faithfulness", "faithful", 10),
        ("citation_accuracy", "cite_acc", 10), ("false_refusal", "false_ref", 11),
        ("hallucinated", "halluc", 8), ("overall_score", "score", 7)]


def _fmt(key: str, value) -> str:
    if value is None:
        return "-"
    return str(value) if key == "n" else f"{value:.2f}"


def print_summary(summary: dict) -> None:
    print("\n" + " " * 13 + "".join(f"{label:>{w}}" for _, label, w in COLS))
    for name, s in summary.items():
        print(f"{name:<13}" + "".join(f"{_fmt(k, s.get(k)):>{w}}" for k, _, w in COLS))


# ------------------------------------------------------------------ indexing for the eval run
def _corpus_signature(strategy: str) -> str:
    h = hashlib.sha1()
    for p in sorted(CORPUS_DIR.glob("*")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return f"{strategy}|{CHUNK_SIZE}|{CHUNK_OVERLAP}|{EMBEDDING_PROVIDER}|{h.hexdigest()[:12]}"


def _index_stats() -> dict:
    sizes = [len(r["text"]) for r in load_chunk_rows()]
    dups = 0
    if DUPLICATES_FILE.exists():
        dups = sum(1 for line in DUPLICATES_FILE.read_text(encoding="utf-8").splitlines()
                   if line.strip())
    return {"chunks": len(sizes),
            "avg_chars": round(sum(sizes) / len(sizes), 1) if sizes else 0,
            "min_chars": min(sizes) if sizes else 0,
            "max_chars": max(sizes) if sizes else 0,
            "duplicates_skipped": dups}


def prepare_index(strategy: str, force: bool) -> dict:
    """Index the eval corpus into this run's own data folder (never touches your real index)."""
    meta_file = DATA_DIR / "run_meta.json"
    signature = _corpus_signature(strategy)
    current = None
    if meta_file.exists():
        current = json.loads(meta_file.read_text(encoding="utf-8")).get("signature")

    if force or current != signature or get_collection().count() == 0:
        print(f"Indexing eval corpus: strategy={strategy}, chunk_size={CHUNK_SIZE}, "
              f"overlap={CHUNK_OVERLAP}")
        ingest_path(CORPUS_DIR, strategy, reset=True)
        stats = {"signature": signature, **_index_stats()}
        meta_file.write_text(json.dumps(stats), encoding="utf-8")
    else:
        print("Eval index is up to date, reusing it.")
    return json.loads(meta_file.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ helpers
def _sample(cases: list[dict], n: int) -> list[dict]:
    """Take n cases round-robin across question types, so a small run still covers every type."""
    by_type: dict[str, list[dict]] = {}
    for c in cases:
        by_type.setdefault(c["type"], []).append(c)
    picked: list[dict] = []
    while len(picked) < n and any(by_type.values()):
        for t in list(by_type):
            if by_type[t] and len(picked) < n:
                picked.append(by_type[t].pop(0))
    return picked


def compare_to_baseline(summary: dict, baseline_file: str, max_drop: float) -> list[str]:
    base = json.loads(Path(baseline_file).read_text(encoding="utf-8"))["summary"]["overall"]
    print(f"\nChange versus baseline ({baseline_file}):")
    regressions = []
    for m in HEADLINE + ["overall_score"]:
        new, old = summary["overall"].get(m), base.get(m)
        if new is None or old is None:
            continue
        delta = new - old
        flag = "  <-- REGRESSION" if delta < -max_drop else ""
        print(f"  {m:<20}{old:>6.3f} -> {new:>6.3f}  ({delta:+.3f}){flag}")
        if delta < -max_drop:
            regressions.append(m)
    return regressions


# ------------------------------------------------------------------ main entry
def run(args) -> int:
    cases = json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))
    if args.types:
        wanted = {t.strip() for t in args.types.split(",")}
        cases = [c for c in cases if c["type"] in wanted]
    if args.limit:
        cases = _sample(cases, args.limit)
    if not cases:
        print("No test cases selected.")
        return 1

    stats = prepare_index(args.strategy, args.ingest)
    retriever = Retriever()
    retriever.search("warm up")                    # load the models before timing starts

    print(f"\nRunning {len(cases)} cases "
          f"({'retrieval only' if args.retrieval_only else 'full pipeline + judge'})\n")
    rows = []
    for i, case in enumerate(cases, 1):
        print(f"[{i:>2}/{len(cases)}] {case['id']} ({case['type']}) {case['question'][:58]}")
        rows.append(run_case(case, retriever, args.retrieval_only, not args.no_faithfulness))

    summary = summarize(rows)
    print_summary(summary)

    bad = [r for r in rows if r.get("answer_correctness") == 0.0][:8]
    if bad:
        print("\nCases to look at (correctness 0):")
        for r in bad:
            why = r.get("correctness_reason") or (r.get("answer") or "")[:90]
            print(f"  {r['id']}: {r['question'][:55]}\n        {why}")

    result = {
        "meta": {"strategy": args.strategy, "chunk_size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP,
                 "embedding": EMBEDDING_PROVIDER, "llm": LLM_MODEL, "judge": JUDGE_MODEL,
                 "min_retrieval_confidence": MIN_RETRIEVAL_CONFIDENCE,
                 "retrieval_only": bool(args.retrieval_only), "n_cases": len(rows),
                 "index": stats, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
        "summary": summary, "cases": rows}
    out = Path(args.out) if args.out else RESULTS_DIR / f"{args.strategy}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {out}")

    if args.baseline:
        regressions = compare_to_baseline(summary, args.baseline, args.max_drop)
        if regressions:
            print(f"\nFAILED: {', '.join(regressions)} dropped by more than {args.max_drop}.")
            return 1
        print("\nNo regressions beyond the allowed drop.")
    return 0
