
from __future__ import annotations

import argparse

from app.config import DENSE_WEIGHT, TOP_K
from app.retrieval import Retriever


def _fmt(v, nd=3):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def main():
    ap = argparse.ArgumentParser(description="Hybrid search over the indexed docs")
    ap.add_argument("query")
    ap.add_argument("--mode", choices=["dense", "sparse", "hybrid"], default="hybrid")
    ap.add_argument("--k", type=int, default=TOP_K)
    ap.add_argument("--dense-weight", type=float, default=DENSE_WEIGHT,
                    help="0.7 means 70%% dense / 30%% sparse")
    ap.add_argument("--no-rerank", action="store_true")
    args = ap.parse_args()

    hits = Retriever().search(args.query, mode=args.mode, k=args.k,
                              dense_weight=args.dense_weight, rerank=not args.no_rerank)

    if not hits:
        print("No results.")
        return
    for i, h in enumerate(hits, 1):
        where = f"{h.source} > {h.heading or '-'}" + (f" (p.{h.page})" if h.page else "")
        print(f"\n#{i}  {where}")
        print(f"    dense_rank={_fmt(h.dense_rank)}  sparse_rank={_fmt(h.sparse_rank)}  "
              f"rrf={_fmt(h.rrf_score, 5)}  rerank={_fmt(h.rerank_score)}")
        print("    " + h.text[:220].replace("\n", " "))


if __name__ == "__main__":
    main()