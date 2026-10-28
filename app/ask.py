
from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from app.generation import answer_question

ICON = {"supported": "OK  ", "partial": "PART", "unsupported": "FAIL",
        "uncited": "NONE", "invalid_reference": "BAD ", "gap": "GAP "}


def main():
    ap = argparse.ArgumentParser(description="Ask a question about the indexed documents")
    ap.add_argument("question")
    ap.add_argument("--no-verify", action="store_true", help="skip citation checking (faster)")
    ap.add_argument("--json", action="store_true", help="print the raw structured response")
    args = ap.parse_args()

    r = answer_question(args.question, verify=not args.no_verify)

    if args.json:
        print(json.dumps(asdict(r), indent=2, ensure_ascii=False))
        return

    print(f"\nQuestion: {r.question}")
    print(f"Retrieval confidence: {r.retrieval_confidence:.2f}")

    if r.status == "insufficient_context":
        print("\nI couldn't find a reliable answer in the documents.\n")
        print("What I found:     ", r.found)
        print("What I couldn't find:", r.not_found)
        print("Worth checking manually:", ", ".join(r.suggested_documents) or "-")
        return

    print("\n--- ANSWER ---")
    print(r.answer)

    print("\n--- SOURCES ---")
    for s in r.sources:
        where = s["source"] + (f" > {s['heading']}" if s["heading"] else "")
        print(f"[{s['ref']}] {where}")

    if r.citation_checks:
        print("\n--- CITATION CHECKS ---")
        for c in r.citation_checks:
            print(f"[{ICON.get(c.verdict, c.verdict)}] {c.claim}  {c.citations}\n         {c.reason}")

    if r.confidence:
        c = r.confidence
        print("\n--- CONFIDENCE ---")
        print(f"retrieval={c.retrieval:.2f}  citation_coverage={c.citation_coverage:.2f}  "
              f"completeness={c.completeness:.2f}  -> composite={c.composite:.2f}")

    for w in r.warnings:
        print("WARNING:", w)


if __name__ == "__main__":
    main()