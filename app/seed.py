"""Index the bundled sample documents so a fresh install can answer questions immediately.

    python -m app.seed            # only if the index is empty or out of sync
    python -m app.seed --force    # wipe everything and re-index the samples
"""
from __future__ import annotations

import argparse
import os
import sys

from app.config import ROOT
from app.indexing import get_collection, verify_sync
from app.ingest import ingest_path

SAMPLE_DIR = ROOT / "sample_docs"


def main() -> None:
    ap = argparse.ArgumentParser(description="Index the sample documentation corpus")
    ap.add_argument("--force", action="store_true", help="wipe the indexes and re-seed")
    args = ap.parse_args()

    strategy = os.getenv("SEED_STRATEGY", "recursive")
    collection = get_collection()

    if not args.force and collection.count() > 0 and verify_sync(collection):
        print(f"Index already populated ({collection.count()} chunks). Nothing to seed. "
              f"Use --force to rebuild.")
        return

    if not SAMPLE_DIR.exists() or not any(SAMPLE_DIR.iterdir()):
        sys.exit(f"No sample documents found in {SAMPLE_DIR}")

    print(f"Seeding the index from {SAMPLE_DIR} (strategy={strategy}) ...")
    ingest_path(SAMPLE_DIR, strategy, reset=True)
    print("Seeding finished.")


if __name__ == "__main__":
    main()
