"""Command-line entry point for the evaluation suite.

    python -m app.evaluate --strategy recursive --retrieval-only     # fast, free, no LLM
    python -m app.evaluate --strategy recursive --limit 8            # small smoke test
    python -m app.evaluate --strategy recursive                      # full run
    python -m app.evaluate --strategy recursive --baseline eval/baseline.json

This file only uses the standard library on purpose: it re-launches itself with
RAG_DATA_DIR pointing at eval/runs/<strategy>, so the evaluation builds and uses its own
index and NEVER touches your real data/ folder.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STRATEGIES = ["fixed", "recursive", "semantic"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Run the RAG evaluation suite")
    ap.add_argument("--strategy", choices=STRATEGIES, default="recursive")
    ap.add_argument("--chunk-size", type=int, default=500,
                    help="characters per chunk during evaluation (small, so the strategies differ)")
    ap.add_argument("--overlap", type=int, default=100)
    ap.add_argument("--retrieval-only", action="store_true",
                    help="only retrieval metrics: no LLM calls at all")
    ap.add_argument("--no-faithfulness", action="store_true",
                    help="skip the faithfulness judge (fewer LLM calls)")
    ap.add_argument("--limit", type=int, help="run only N cases, spread across question types")
    ap.add_argument("--types", help="comma list: lookup,multi_hop,no_answer,ambiguous")
    ap.add_argument("--ingest", action="store_true", help="force re-indexing of the eval corpus")
    ap.add_argument("--no-cache", action="store_true", help="do not reuse cached LLM calls")
    ap.add_argument("--out", help="where to save the JSON results")
    ap.add_argument("--baseline", help="earlier results JSON to compare against")
    ap.add_argument("--max-drop", type=float, default=0.05,
                    help="exit with code 1 if a headline metric drops by more than this")
    return ap.parse_args(argv)


def main() -> None:
    args = parse_args()

    if "RAG_DATA_DIR" not in os.environ:
        env = os.environ.copy()
        env["RAG_DATA_DIR"] = str(ROOT / "eval" / "runs" / args.strategy)
        env["CHUNK_SIZE"] = str(args.chunk_size)
        env["CHUNK_OVERLAP"] = str(args.overlap)
        if not args.no_cache and not args.retrieval_only:
            env["LLM_CACHE"] = "1"
        code = subprocess.run([sys.executable, "-m", "app.evaluate", *sys.argv[1:]],
                              env=env, cwd=ROOT).returncode
        sys.exit(code)

    from app.eval_suite import run          # imported late, after RAG_DATA_DIR is set
    sys.exit(run(args))


if __name__ == "__main__":
    main()
