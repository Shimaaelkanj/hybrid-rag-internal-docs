"""Run the eval suite once per chunking strategy and write a comparison report.

    python -m app.compare                       # full run for fixed, recursive, semantic
    python -m app.compare --retrieval-only      # fast, no LLM
    python -m app.compare --limit 12            # smaller run
    python -m app.compare --reuse               # rebuild the report from saved results

Any other option (--chunk-size, --overlap, --no-faithfulness, ...) is passed on to
app.evaluate. Output: eval/results/<strategy>.json and eval/report.md
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "eval" / "results"
REPORT_FILE = ROOT / "eval" / "report.md"
STRATEGIES = ["fixed", "recursive", "semantic"]

# (metric key, column title, lower is better, decimals)
OVERALL_COLUMNS = [
    ("retrieval_recall", "Recall", False, 2), ("retrieval_mrr", "MRR", False, 2),
    ("context_precision", "Ctx precision", False, 2), ("answer_correctness", "Correctness", False, 2),
    ("faithfulness", "Faithfulness", False, 2), ("citation_accuracy", "Citation acc.", False, 2),
    ("false_refusal", "False refusal", True, 2), ("hallucinated", "Hallucinated", True, 2),
    ("overall_score", "Score", False, 2), ("latency_s", "Latency (s)", True, 1),
]
TYPE_COLUMNS = [
    ("retrieval_recall", "Recall", False, 2), ("answer_correctness", "Correctness", False, 2),
    ("faithfulness", "Faithfulness", False, 2), ("citation_accuracy", "Citation acc.", False, 2),
]
WINNER_METRICS = [("retrieval_recall", "Retrieval recall"), ("context_precision", "Context precision"),
                  ("answer_correctness", "Answer correctness"), ("faithfulness", "Faithfulness"),
                  ("citation_accuracy", "Citation accuracy"), ("overall_score", "Overall score")]


def _fmt(value, decimals: int) -> str:
    return "-" if value is None else f"{value:.{decimals}f}"


def _table(rows: dict[str, dict], columns) -> list[str]:
    """Markdown table, one row per strategy; the best value in each column is bold."""
    lines = ["| Strategy | " + " | ".join(c[1] for c in columns) + " |",
             "|---|" + "---|" * len(columns)]
    best: dict[str, set] = {}
    for key, _, lower, _ in columns:
        values = {s: r.get(key) for s, r in rows.items() if r.get(key) is not None}
        if values and len(set(values.values())) > 1:          # no bold when everything ties
            target = min(values.values()) if lower else max(values.values())
            best[key] = {s for s, v in values.items() if v == target}
    for strategy, summary in rows.items():
        cells = []
        for key, _, _, decimals in columns:
            text = _fmt(summary.get(key), decimals)
            cells.append(f"**{text}**" if strategy in best.get(key, set()) else text)
        lines.append(f"| {strategy} | " + " | ".join(cells) + " |")
    return lines


def build_report(results: dict[str, dict]) -> str:
    strategies = list(results)
    first = results[strategies[0]]["meta"]
    out = ["# Chunking strategy comparison", "",
           f"Generated {time.strftime('%Y-%m-%d %H:%M')}. Bold = best value in the column.", "",
           "## Setup", "",
           f"- Test cases: {first['n_cases']} (same questions for every strategy)",
           f"- Chunk size / overlap: {first['chunk_size']} / {first['overlap']} characters",
           f"- Embeddings: {first['embedding']}, generator: {first['llm']}, judge: {first['judge']}",
           f"- Mode: {'retrieval only (no LLM)' if first['retrieval_only'] else 'full pipeline with LLM judge'}",
           "", "| Strategy | Chunks | Avg chars | Min | Max | Duplicates skipped |", "|---|---|---|---|---|---|"]
    for s in strategies:
        ix = results[s]["meta"]["index"]
        out.append(f"| {s} | {ix['chunks']} | {ix['avg_chars']} | {ix['min_chars']} | "
                   f"{ix['max_chars']} | {ix['duplicates_skipped']} |")

    out += ["", "## Overall", ""]
    out += _table({s: results[s]["summary"]["overall"] for s in strategies}, OVERALL_COLUMNS)

    types = sorted({t for s in strategies for t in results[s]["summary"] if t != "overall"})
    out += ["", "## By question type", ""]
    for t in types:
        present = {s: results[s]["summary"][t] for s in strategies if t in results[s]["summary"]}
        n = next(iter(present.values()))["n"]
        out += [f"### {t} (n={n})", ""] + _table(present, TYPE_COLUMNS) + [""]

    out += ["## Winners", ""]
    for key, label in WINNER_METRICS:
        values = {s: results[s]["summary"]["overall"].get(key) for s in strategies}
        values = {s: v for s, v in values.items() if v is not None}
        if not values:
            continue
        top = max(values.values())
        winners = [s for s, v in values.items() if v == top]
        verdict = "tie between " + ", ".join(winners) if len(winners) > 1 else winners[0]
        out.append(f"- **{label}:** {verdict} ({top:.2f})")

    out += ["", "## How to read this", "",
            "- Recall / context precision measure retrieval; correctness, faithfulness and citation "
            "accuracy are judged by an LLM and are only as reliable as the judge model.",
            "- With a small test set, differences under about 0.05 are within noise. "
            "Add more documents and questions before drawing strong conclusions.",
            "- False refusal = answerable questions the system refused. Hallucinated = unanswerable "
            "questions that received an answer instead of a refusal. Lower is better for both."]
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategies", default=",".join(STRATEGIES))
    ap.add_argument("--reuse", action="store_true", help="skip running, rebuild report from saved results")
    args, passthrough = ap.parse_known_args()
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    if not args.reuse:
        for s in strategies:
            print(f"\n==================== {s} ====================")
            code = subprocess.run([sys.executable, "-m", "app.evaluate", "--strategy", s, *passthrough],
                                  cwd=ROOT).returncode
            if code != 0:
                sys.exit(f"Evaluation failed for strategy '{s}' (exit code {code}).")

    results = {}
    for s in strategies:
        f = RESULTS_DIR / f"{s}.json"
        if f.exists():
            results[s] = json.loads(f.read_text(encoding="utf-8"))
        else:
            print(f"Missing results for {s}: {f}")
    if not results:
        sys.exit("No results found. Run without --reuse first.")

    report = build_report(results)
    REPORT_FILE.write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"Saved {REPORT_FILE}")


if __name__ == "__main__":
    main()
